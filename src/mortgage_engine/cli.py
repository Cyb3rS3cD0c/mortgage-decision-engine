"""Interactive local course demonstration; no unattended human approvals."""
import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langgraph.types import Command
from .workflow import build_workflow, initial_state
from .services import LiveModel, PolicyStore, ServiceFailure

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--case',type=int,default=1,help='One-based fixture index')
    args=parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[2]/'.env',override=True)
    key=os.getenv('OPENAI_API_KEY');base=os.getenv('OPENAI_BASE_URL','https://aibe.mygreatlearning.com/openai/v1')
    if not key:parser.error('Set OPENAI_API_KEY in the ignored local .env')
    cases=json.loads(args.cases.read_text(encoding='utf-8'))['test_cases']
    if not 1<=args.case<=len(cases):parser.error('Case index out of range')
    try:
        graph=build_workflow(LiveModel(base,key),PolicyStore(args.policy,base,key))
        config={'configurable':{'thread_id':'interactive'}}
        state=graph.invoke(initial_state(cases[args.case-1]),config)
        for role,report in state.get('analyses',{}).items():print(role.upper(),report,sep='\n')
        if state.get('__interrupt__'):
            print('Review required:',state['review_reasons'])
            print('Recommendation:',state['recommendation'],'Metrics:',state['metrics'])
            print('Press Enter without a reviewer to leave the application pending.')
            reviewer=input('Reviewer identifier: ').strip()
            if not reviewer:print('PENDING: no reviewer decision recorded');return
            decision=input('Decision [APPROVED/CONDITIONAL_APPROVAL/DENIED]: ').strip().upper()
            notes=input('Review rationale: ').strip()
            state=graph.invoke(Command(resume={'reviewer':reviewer,'decision':decision,'notes':notes,'simulated':False}),config)
        if state.get('status')=='ERROR':print('ERROR:',state['error']);return
        print('Final decision:',state['final_decision'])
        print(state.get('executive_summary',''))
    except (ServiceFailure,ValueError,KeyError) as exc:
        print('Run stopped:',str(exc))
        raise SystemExit(1) from None
if __name__=='__main__':main()
