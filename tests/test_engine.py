
import copy
import pytest
from mortgage_engine.core import *
from mortgage_engine.workflow import build_workflow,initial_state
from mortgage_engine.services import ServiceFailure
from langgraph.types import Command

def sample():
    return {'case_id':'SYNTHETIC-01','name':'Example Borrower','ssn':'111-22-3333',
      'email':'example@example.invalid','credit_score':760,'dti_ratio':0.3,
      'employment':{'monthly_income':10000},'debts':{'car':500,'total_monthly_debt':500},
      'assets':{'checking':100000,'savings':50000},
      'loan':{'amount':240000,'down_payment':60000,'closing_costs':5000,'estimated_payment':2000,'property_type':'Single Family'},
      'property':{'appraised_value':300000,'required_repairs':0}}
class FakeStore:
    def retrieve(self,q):return [{'source':'synthetic policy','page':1,'text':'Test evidence only'}]
class FakeModel:
    def __init__(self):self.payloads=[]
    def analyze(self,role,payload,evidence):
        self.payloads.append(payload)
        return 'Synthetic analysis with test evidence; no real model call.'
class FailedModel:
    def analyze(self,*args):raise ServiceFailure('Simulated service outage')

def test_debt_total_not_duplicated():
    assert metrics(sample())['total_debt']==2500
    assert metrics(sample())['dti_percent']==25

def test_inconsistent_total_rejected():
    a=sample();a['debts']['total_monthly_debt']=999
    with pytest.raises(ValueError):metrics(a)

@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf'),True])
def test_invalid_income(value):
    with pytest.raises(ValueError):calculate_dti_ratio.invoke({'monthly_debt':100,'monthly_income':value})

def test_postclosing_reserves():
    r=calculate_reserves.invoke({'liquid_assets':22000,'monthly_payment':2400,'down_payment':35000,'closing_costs':6000})
    assert r['reserve_months']==0 and r['cash_to_close_shortfall']==19000

def test_sanitization_deep_and_narrative():
    a=sample();a['nested']={'phone':'(415) 555-9876','notes':'Example Borrower example@example.invalid 111-22-3333 (415) 555-9876'}
    original=copy.deepcopy(a);safe=sanitize_pii(a)
    for token in ['Example Borrower','111-22-3333','example@example.invalid','555-9876']:assert token not in str(safe)
    assert a==original

def test_bias_word_boundaries():
    assert detect_bias_signals('Mortgage coverage and property collateral')==[]
    assert detect_bias_signals('Reject due to religion')

def test_deposit_threshold_and_aggregate():
    r=check_large_deposits.invoke({'deposits':[{'amount':600},{'amount':500}],'monthly_income':10000})
    assert r['threshold']==1000 and r['aggregate_review'] and not r['large_deposits']

def test_no_labels_or_identifiers_in_graph():
    a=sample();a['expected_decision']='APPROVED'
    s=initial_state(a)
    assert 'expected_decision' not in s['application']
    assert '111-22-3333' not in str(s)

def test_low_risk_completes_without_review():
    model=FakeModel();g=build_workflow(model,FakeStore())
    r=g.invoke(initial_state(sample()),{'configurable':{'thread_id':'low'}})
    assert r['status']=='COMPLETE' and r['final_decision']=='APPROVED'
    assert list(r['analyses'])==['credit','income','asset','collateral','critic','decision']

def test_actual_interrupt_and_resume():
    a=sample();a['employment']['monthly_income']=3000
    g=build_workflow(FakeModel(),FakeStore());c={'configurable':{'thread_id':'high'}}
    r=g.invoke(initial_state(a),c)
    assert r['__interrupt__'] and r['final_decision']=='PENDING'
    r=g.invoke(Command(resume={'reviewer':'TEST','notes':'Simulated review of unresolved DTI failure','decision':'DENIED','simulated':True}),c)
    assert r['human_review_completed'] and r['final_decision']=='DENIED'

def test_hard_failure_override_rejected():
    a=sample();a['credit_score']=595
    g=build_workflow(FakeModel(),FakeStore());c={'configurable':{'thread_id':'override'}}
    g.invoke(initial_state(a),c)
    with pytest.raises(ValueError):g.invoke(Command(resume={'reviewer':'TEST','notes':'Test override','decision':'APPROVED'}),c)

def test_service_failure_stops_without_decision():
    g=build_workflow(FailedModel(),FakeStore())
    r=g.invoke(initial_state(sample()),{'configurable':{'thread_id':'failure'}})
    assert r['status']=='ERROR' and r['final_decision']=='PENDING'
    assert 'decision' not in r['analyses']

def test_thresholds_strictly_greater():
    a=sample();a['loan']['amount']=270000
    a['employment']['monthly_income']=2500/.43
    g=build_workflow(FakeModel(),FakeStore())
    r=g.invoke(initial_state(a),{'configurable':{'thread_id':'boundary'}})
    assert r['metrics']['dti_percent']==43 and r['metrics']['ltv_percent']==90
    assert not r['human_review_required']


def test_missing_policy_evidence_stops():
    class EmptyStore:
        def retrieve(self,q):return []
    graph=build_workflow(FakeModel(),EmptyStore())
    result=graph.invoke(initial_state(sample()),{'configurable':{'thread_id':'empty-policy'}})
    assert result['status']=='ERROR' and result['final_decision']=='PENDING'

def test_bias_requires_review_even_low_score():
    class BiasedModel:
        def analyze(self,*args):return 'Decision uses religion. Synthetic negative test.'
    graph=build_workflow(BiasedModel(),FakeStore())
    result=graph.invoke(initial_state(sample()),{'configurable':{'thread_id':'bias'}})
    assert result['risk_score']==0 and result['bias_flags'] and result['__interrupt__']

def test_review_requires_rationale():
    a=sample();a['credit_score']=595
    graph=build_workflow(FakeModel(),FakeStore());cfg={'configurable':{'thread_id':'missing-notes'}}
    graph.invoke(initial_state(a),cfg)
    with pytest.raises(ValueError):graph.invoke(Command(resume={'reviewer':'TEST','decision':'DENIED','notes':''}),cfg)
