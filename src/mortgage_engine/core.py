"""Deterministic tools and data protection. No network calls."""
import copy
import math
import re
from typing import Any
from pydantic import StrictFloat, StrictInt
from langchain_core.tools import tool

def number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)

@tool
def calculate_dti_ratio(monthly_debt: StrictFloat, monthly_income: StrictFloat) -> dict:
    """Return back-end debt-to-income percentage, including housing once."""
    return {"dti_percent": round(100*number(monthly_debt,'debt')/number(monthly_income,'income',True),4)}

@tool
def calculate_ltv_ratio(loan_amount: StrictFloat, property_value: StrictFloat) -> dict:
    """Return appraised loan-to-value percentage."""
    return {"ltv_percent": round(100*number(loan_amount,'loan')/number(property_value,'value',True),4)}

@tool
def calculate_housing_expense_ratio(monthly_payment: StrictFloat, monthly_income: StrictFloat) -> dict:
    """Return housing expense percentage."""
    return {"housing_percent": round(100*number(monthly_payment,'payment')/number(monthly_income,'income',True),4)}

@tool
def calculate_total_debt_obligations(debts: dict, proposed_payment: StrictFloat) -> dict:
    """Sum itemized debts once; compare, but do not add, aggregate total."""
    items={k:number(v,k) for k,v in debts.items() if k!='total_monthly_debt'}
    if not items:
        raise ValueError('Itemized debts are required')
    total=sum(items.values())
    supplied=debts.get('total_monthly_debt')
    if supplied is not None and abs(number(supplied,'debt total')-total)>0.01:
        raise ValueError('Itemized debts disagree with debt total')
    return {"existing_debt":total,"housing_payment":number(proposed_payment,'payment',True),
            "total_debt":total+proposed_payment}

@tool
def calculate_reserves(liquid_assets: StrictFloat, monthly_payment: StrictFloat, down_payment: StrictFloat=0,
                       closing_costs: StrictFloat=0, required_months: StrictInt=2) -> dict:
    """Calculate reserves after cash to close; preserve shortages as deficits."""
    remaining=number(liquid_assets,'assets')-number(down_payment,'down payment')-number(closing_costs,'closing costs')
    payment=number(monthly_payment,'payment',True)
    number(required_months,'required months',True)
    return {"cash_after_closing":remaining,"reserve_months":round(max(0,remaining)/payment,4),
            "cash_to_close_shortfall":max(0,-remaining),"adequate":remaining>=payment*required_months}

@tool
def check_credit_score_policy(credit_score: StrictInt) -> dict:
    """Evaluate educational conventional-loan score minimum of 620."""
    number(credit_score,'credit score')
    if not 300<=credit_score<=850 or int(credit_score)!=credit_score:
        raise ValueError('Credit score must be an integer from 300 to 850')
    return {"score":credit_score,"minimum":620,"meets_minimum":credit_score>=620}

@tool
def check_large_deposits(deposits: list, monthly_income: StrictFloat) -> dict:
    """Flag deposits above min($1000, 25% income), and aggregate >=$1000."""
    threshold=min(1000,number(monthly_income,'income',True)*0.25)
    values=[number(d.get('amount'),'deposit') for d in deposits]
    return {"threshold":threshold,"large_deposits":[{"index":i,"amount":v} for i,v in enumerate(values) if v>threshold],
            "aggregate_review":sum(values)>=1000,
            "note":"Flag requires sourcing review; a documented payroll deposit is not automatically adverse."}

SENSITIVE={'name','ssn','email','phone','address','employer','zip','zipcode','date_of_birth','dob',
           'race','religion','sex','gender','marital_status','national_origin','disability','age'}

def sanitize_pii(data: dict) -> dict:
    """Deep-redact structured identifiers and repeated identifiers in narrative text."""
    tokens=[]
    def collect(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k.lower() in SENSITIVE and isinstance(v,str) and len(v)>2: tokens.append(v)
                collect(v)
        elif isinstance(x,list):
            for v in x: collect(v)
    collect(data)
    def walk(x):
        if isinstance(x,dict):return {k:'[REDACTED]' if k.lower() in SENSITIVE else walk(v) for k,v in x.items()}
        if isinstance(x,list):return [walk(v) for v in x]
        if isinstance(x,str):
            for token in sorted(set(tokens),key=len,reverse=True):x=re.sub(re.escape(token),'[REDACTED]',x,flags=re.I)
            x=re.sub(r'\b\d{3}-\d{2}-\d{4}\b','[REDACTED]',x)
            x=re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}','[REDACTED]',x)
            x=re.sub(r'(?<!\d)(?:\+1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\d)','[REDACTED]',x)
        return copy.deepcopy(x)
    return walk(data)

def detect_bias_signals(analysis: str, applicant_data: dict|None=None) -> list[str]:
    """Flag mentions for human inspection, not a legal finding or fairness proof."""
    terms=['race','color','religion','national origin','sex','marital status','age','gender','disability','familial status']
    flags=[f'Protected-characteristic mention: {t}' for t in terms if re.search(r'\b'+re.escape(t)+r'\b',analysis,re.I)]
    if re.search(r'\b(zip\s*code|neighborhood)\b',analysis,re.I): flags.append('Potential geographic proxy mention')
    return flags

def metrics(app):
    loan=app['loan'];income=app['employment']['monthly_income']
    debts=calculate_total_debt_obligations.invoke({'debts':app['debts'],'proposed_payment':loan['estimated_payment']})
    result={**debts,**calculate_dti_ratio.invoke({'monthly_debt':debts['total_debt'],'monthly_income':income}),
        **calculate_ltv_ratio.invoke({'loan_amount':loan['amount'],'property_value':app['property']['appraised_value']}),
        **calculate_housing_expense_ratio.invoke({'monthly_payment':loan['estimated_payment'],'monthly_income':income}),
        **calculate_reserves.invoke({'liquid_assets':app['assets']['checking']+app['assets']['savings'],
        'monthly_payment':loan['estimated_payment'],'down_payment':loan['down_payment'],'closing_costs':loan['closing_costs']}),
        **check_credit_score_policy.invoke({'credit_score':app['credit_score']})}
    supplied=number(app['dti_ratio'],'supplied DTI')*100
    result['supplied_dti_percent']=supplied
    result['dti_discrepancy']=abs(supplied-result['dti_percent'])>0.01
    return result

def risk_assessment(app,m):
    """Transparent educational score; hard failures override score bands."""
    credit=25 if m['score']<620 else 15 if m['score']<700 else 0
    income=25 if m['dti_percent']>50 else 18 if m['dti_percent']>43 else 8 if m['housing_percent']>28 else 0
    assets=25 if m['cash_to_close_shortfall'] else 15 if m['reserve_months']<2 else 0
    repair=number(app['property']['required_repairs'],'repairs')
    collateral=25 if m['ltv_percent']>97 or repair>5000 else 12 if repair or m['ltv_percent']>90 else 0
    violations=[];conditions=[]
    if m['score']<620:violations.append('Credit score below conventional minimum 620 (policy 1.1)')
    if m['dti_percent']>50:violations.append('DTI above maximum compensating-factor ceiling 50% (policy 2.6)')
    elif m['dti_percent']>43:conditions.append('Senior review of compensating factors for DTI (policy 2.6)')
    if m['cash_to_close_shortfall']:violations.append('Insufficient cash to close (policy 3.1)')
    if m['reserve_months']<2:conditions.append('Verify at least two months post-closing reserves (policy 3.1)')
    if repair:conditions.append('Verify required repairs and habitability before closing (policy 4.2)')
    if app['loan']['property_type']=='Condo':conditions.append('Condominium project eligibility documentation required (policy 4.6)')
    score=credit+income+assets+collateral
    decision='DENIED' if violations or score>70 else 'CONDITIONAL_APPROVAL' if conditions or score>=30 else 'APPROVED'
    return {'risk_breakdown':dict(credit=credit,income=income,assets=assets,collateral=collateral),
            'risk_score':score,'recommendation':decision,'policy_violations':violations,'conditions':conditions}

