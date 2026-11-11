"""Sequential specialist orchestration with a real interrupt/resume boundary."""
import operator
from datetime import datetime, timezone
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
from .core import sanitize_pii,metrics,risk_assessment,detect_bias_signals
from .services import ServiceFailure

class UnderwritingState(TypedDict,total=False):
    case_id:str
    application:dict
    metrics:dict
    analyses:dict
    evidence:dict
    next_agent:str
    risk_score:int
    risk_breakdown:dict
    recommendation:str
    policy_violations:list
    conditions:list
    bias_flags:list
    review_reasons:list
    human_review_required:bool
    human_review_completed:bool
    human_review:dict
    final_decision:str
    status:str
    error:str
    decision_memo:str
    executive_summary:str
    audit:Annotated[list,operator.add]

def event(node,detail):
    return [{'node':node,'detail':detail,'timestamp':datetime.now(timezone.utc).isoformat()}]

QUERIES={
 'credit':'credit score minimum bankruptcy late payments collections',
 'income':'employment income verification DTI front end back end compensating factors',
 'asset':'assets reserves after closing down payment large deposits documentation',
 'collateral':'appraisal property condition repair requirements LTV condominium eligibility',
 'critic':'credit income assets property policy compliance consistency review',
 'decision':'DTI compensating factors reserves credit minimum exception approval'}

ROLES={
 'credit':'Credit Analyst: score, payment history, bankruptcies, collections and missing discharge evidence',
 'income':'Income Analyst: employment stability, income documentation, exact DTI and housing ratios, gaps and variable income',
 'asset':'Asset Analyst: cash to close, post-closing reserves, large deposit sourcing and missing evidence',
 'collateral':'Collateral Analyst: appraisal LTV, habitability, repair details, condo eligibility and missing documentation',
 'critic':'Critic: check completeness, contradictions, data discrepancies and policy support across all four specialist reports',
 'decision':'Senior Underwriter: explain the supplied deterministic recommendation, conditions, risk score and reasons; do not override hard policy failures'}

def build_workflow(model,store):
    def initialize(state):
        try:
            m=metrics(state['application'])
            return {'metrics':m,'analyses':{},'evidence':{},'status':'ANALYZING',
                'human_review_completed':False,'audit':event('initialize','Validated data; recalculated DTI without aggregate duplication')}
        except (ValueError,KeyError,TypeError):
            return {'status':'ERROR','error':'Invalid or incomplete application data','final_decision':'PENDING',
                'audit':event('initialize','Input validation failed; no decision issued')}
    def supervisor(state):
        next_agent=next((r for r in ('credit','income','asset','collateral') if r not in state['analyses']),'critic')
        return {'next_agent':next_agent,'audit':event('supervisor','Route to '+next_agent)}
    def analysis_node(role):
        def run(state):
            try:
                evidence=store.retrieve(QUERIES[role])
                if not evidence:raise ServiceFailure('No policy evidence retrieved; application remains pending')
                payload={'application':state['application'],'metrics':state['metrics']}
                if role in ('critic','decision'):payload['analyses']=state['analyses']
                if role=='decision':payload['assessment']=risk_assessment(state['application'],state['metrics'])
                report=model.analyze(ROLES[role],payload,evidence)
                # Defensive redaction for identifier-like output content.
                report=sanitize_pii({'text':report})['text']
                return {'analyses':{**state['analyses'],role:report},
                    'evidence':{**state['evidence'],role:evidence},'audit':event(role,'Analysis completed with retrieved policy evidence')}
            except ServiceFailure as exc:
                return {'status':'ERROR','error':str(exc),'final_decision':'PENDING',
                    'audit':event(role,'External service failure; workflow stopped without decision')}
        return run
    def assess(state):
        assessment=risk_assessment(state['application'],state['metrics'])
        flags=[]
        for role,text in state['analyses'].items():flags.extend(f'{role}: {f}' for f in detect_bias_signals(text))
        m=state['metrics'];reasons=[]
        if assessment['risk_score']>30:reasons.append('Risk score exceeds 30')
        if m['dti_percent']>43:reasons.append('DTI exceeds 43%')
        if m['ltv_percent']>90:reasons.append('LTV exceeds 90%')
        if flags:reasons.append('Bias signals require inspection')
        if assessment['recommendation']=='DENIED':reasons.append('Denial requires senior review')
        return {**assessment,'bias_flags':flags,'review_reasons':reasons,'human_review_required':bool(reasons),
            'final_decision':'PENDING' if reasons else assessment['recommendation'],
            'status':'AWAITING_REVIEW' if reasons else 'COMPLETE',
            'decision_memo':state['analyses']['decision'],'audit':event('assess','Applied documented policy gates and review thresholds')}
    def human_review(state):
        review=interrupt({'case_id':state['case_id'],'recommendation':state['recommendation'],
                          'reasons':state['review_reasons'],'metrics':state['metrics']})
        if not isinstance(review,dict) or review.get('decision') not in ('APPROVED','CONDITIONAL_APPROVAL','DENIED'):
            raise ValueError('Review must include a valid decision')
        if not all(isinstance(review.get(k),str) and review[k].strip() for k in ('reviewer','notes')):
            raise ValueError('Reviewer and explanatory notes required')
        if state['policy_violations'] and review['decision']!='DENIED':
            raise ValueError('Unresolved hard policy failures cannot be approved; correct evidence and rerun')
        safe=sanitize_pii(review)
        return {'human_review':safe,'human_review_completed':True,'final_decision':review['decision'],
            'status':'COMPLETE','audit':event('human_review','Review recorded; simulation='+str(review.get('simulated',False)))}
    def summary(state):
        try:
            text=model.analyze('Executive summary: summarize final outcome, metrics, conditions and limitations in 4-6 sentences',
                {'case_id':state['case_id'],'decision':state['final_decision'],'metrics':state['metrics'],
                 'review':state.get('human_review'),'memo':state['decision_memo']},state['evidence']['decision'])
            return {'executive_summary':sanitize_pii({'text':text})['text'],'audit':event('summary','Final report generated')}
        except ServiceFailure as exc:
            return {'status':'ERROR','error':str(exc),'audit':event('summary','Summary unavailable; decision history preserved')}
    graph=StateGraph(UnderwritingState)
    graph.add_node('initialize',initialize);graph.add_node('supervisor',supervisor)
    for role in ROLES:graph.add_node(role,analysis_node(role))
    graph.add_node('assess',assess);graph.add_node('human_review',human_review);graph.add_node('summary',summary)
    graph.add_edge(START,'initialize')
    graph.add_conditional_edges('initialize',lambda s:'stop' if s['status']=='ERROR' else 'go',{'stop':END,'go':'supervisor'})
    graph.add_conditional_edges('supervisor',lambda s:s['next_agent'],{r:r for r in ('credit','income','asset','collateral','critic')})
    for role in ('credit','income','asset','collateral'):
        graph.add_conditional_edges(role,lambda s:'stop' if s['status']=='ERROR' else 'go',{'stop':END,'go':'supervisor'})
    for role,dest in [('critic','decision'),('decision','assess')]:
        graph.add_conditional_edges(role,lambda s:'stop' if s['status']=='ERROR' else 'go',{'stop':END,'go':dest})
    graph.add_conditional_edges('assess',lambda s:'review' if s['human_review_required'] else 'done',{'review':'human_review','done':'summary'})
    graph.add_edge('human_review','summary');graph.add_edge('summary',END)
    return graph.compile(checkpointer=InMemorySaver())

def initial_state(raw):
    # Raw identifiers never enter graph checkpoints or LLM requests.
    safe=sanitize_pii(raw)
    # Expected labels are evaluation targets, not agent inputs.
    safe.pop('expected_decision',None)
    return {'case_id':safe['case_id'],'application':safe,'audit':[]}
