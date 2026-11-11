"""Build, execute and export a private, self-contained grading notebook."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from dotenv import dotenv_values

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO.parent
OUT=ROOT/'submissions'
OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'assignment files'
cells=[]
def md(text):cells.append(nbf.v4.new_markdown_cell(text))
def code(text):cells.append(nbf.v4.new_code_cell(text))
md('''# Senior Mortgage Underwriting System
Johns Hopkins University · Agentic AI · Student: Dr. Jay Lavoie · November 11, 2026

This educational project combines deterministic financial calculations, four specialist LLM analyses, retrieved policy evidence, quality review and human oversight. This notebook contains the complete implementation and bundled private assignment inputs. It requires installed third-party dependencies and a university API credential, but no project source files.

**Private grading artifact.** Do not publish the embedded course material or supplied borrower fixtures. All human-review records below are explicitly simulated demonstrations, not actual underwriting approvals. Policy claims refer to supplied educational material, not independently validated current law.

## Rubric map
| Part | Item | Marks | Evidence section |
|---|---|---:|---|
|1|Utility tools|3|1A|
|1|PII sanitization|2|1B|
|1|Bias detection|2|1B|
|1|RAG policy retrieval|2|1C|
|1|Test preparation|1|1D|
|2|Income, asset, collateral, critic, decision agents|5|2A and 3A full outputs|
|2|State and graph|2|2B|
|3|All test cases|1|3A and 3B|
|4|Summary and observations|1|4A|
|4|Future scope|1|4B|
''')
md('''## Environment and credential setup
Use the project virtual environment, or install the listed packages in a separate environment. Credentials are read from environment variables or a local ignored .env. No credential is embedded. For execution elsewhere, set OPENAI_API_KEY and OPENAI_BASE_URL before running. The configured model is the assignment-required gpt-4o-mini.''')
code('''import os, sys, json, tempfile, base64
from pathlib import Path
from dotenv import load_dotenv
from importlib.metadata import version
local_env = Path(os.environ.get('MORTGAGE_ENV_FILE', '.env'))
if local_env.is_file(): load_dotenv(local_env, override=True)
API_KEY = os.environ.get('OPENAI_API_KEY')
BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://aibe.mygreatlearning.com/openai/v1')
if not API_KEY: raise RuntimeError('Set OPENAI_API_KEY securely before running this notebook')
for package in ['langgraph','langchain-openai','chromadb','pypdf','nbclient','nbconvert']:
    print(package, version(package))
print('Credential loaded: yes (value hidden); model: gpt-4o-mini')
''')
md('## Part 1 Core Agent Implementation\n### 1A Utility tools for calculations\nTools validate finite nonnegative inputs, count each debt once and calculate post-closing reserves. Invalid denominators are rejected.')
core=(REPO/'src/mortgage_engine/core.py').read_text(encoding='utf-8')
code(core)
code('''tool_examples = {
 'DTI': calculate_dti_ratio.invoke({'monthly_debt':2500,'monthly_income':10000}),
 'LTV': calculate_ltv_ratio.invoke({'loan_amount':240000,'property_value':300000}),
 'Housing': calculate_housing_expense_ratio.invoke({'monthly_payment':2000,'monthly_income':10000}),
 'Debts': calculate_total_debt_obligations.invoke({'debts':{'car':500,'total_monthly_debt':500},'proposed_payment':2000}),
 'Reserves': calculate_reserves.invoke({'liquid_assets':10000,'monthly_payment':2000,'down_payment':2000,'closing_costs':1000}),
 'Credit': check_credit_score_policy.invoke({'credit_score':760}),
 'Deposits': check_large_deposits.invoke({'deposits':[{'amount':1500}],'monthly_income':10000})}
print(json.dumps(tool_examples,indent=2))
for bad_income in [0,-1,float('nan'),float('inf')]:
    try: calculate_dti_ratio.invoke({'monthly_debt':100,'monthly_income':bad_income})
    except ValueError: print('Expected validation failure for invalid income:', str(bad_income))
''')
md('### 1B PII sanitization and bias detection\nDeep sanitization covers nested data, email, phone, names and narrative repetitions. Word boundaries prevent “age” matching “mortgage”. Bias signals require review and are not proof of unlawful discrimination or compliance.')
code('''synthetic = {'name':'Example Borrower','ssn':'111-22-3333','email':'sample@example.invalid',
 'nested':{'phone':'(415) 555-9876','note':'Example Borrower at sample@example.invalid'},'income':10000}
safe = sanitize_pii(synthetic)
assert 'Example Borrower' not in str(safe) and 'sample@example.invalid' not in str(safe)
assert synthetic['name']=='Example Borrower'
print('Sanitized synthetic demonstration:',json.dumps(safe,indent=2))
print('Biased wording:',detect_bias_signals('Reject the applicant because of religion'))
print('Neutral wording:',detect_bias_signals('Mortgage coverage and property condition'))
print('Limitation: a neutral result is not a fairness certification.')
''')
md('### 1C RAG policy retrieval\nThe private policy PDF is bundled below to make this notebook independent of local source files. Personalized license footers are removed before embedding. The pipeline splits pages, calls live embeddings, stores them in Chroma, and displays retrieved text with page citations. No silent fallback is used.')
services=(REPO/'src/mortgage_engine/services.py').read_text(encoding='utf-8').replace('from .core import sanitize_pii','')
code(services)
pdf64=base64.b64encode((SOURCE/'underwriting_policies.pdf').read_bytes()).decode()
code("POLICY_PDF_BASE64 = "+repr(pdf64)+"\npolicy_workspace = tempfile.TemporaryDirectory(prefix='mortgage-policy-')\npolicy_path = Path(policy_workspace.name)/'underwriting_policies.pdf'\npolicy_path.write_bytes(base64.b64decode(POLICY_PDF_BASE64))\nprint('Bundled private policy PDF restored for local processing.')")
code('''policy_store = PolicyStore(policy_path,BASE_URL,API_KEY)
print('Live policy chunks indexed:',policy_store.count)
retrieved = policy_store.retrieve('What documentation is required for self-employed borrower income?',k=4)
for item in retrieved:
    print(f"SOURCE: {item['source']} PAGE: {item['page']} DISTANCE: {item['distance']:.4f}")
    print(item['text'])
    print('-'*60)
''')
md('### 1D Test data preparation\nOriginal fixtures are bundled unchanged. Sanitized views are used for all visible reports and model inputs. Supplied labels are retained only for evaluation. The original DTI values and debt aggregation conflict with calculated values; the comparison below exposes the issue.')
data=json.loads((SOURCE/'mortgage_test_cases.json').read_text(encoding='utf-8'))
code('original_test_data = json.loads('+repr(json.dumps(data))+')\ntest_cases = original_test_data["test_cases"]\nprint("Loaded",len(test_cases),"original cases; identifiers suppressed in output.")')
code('''for case in test_cases:
    m=metrics(case)
    print(case['case_id'],json.dumps({'supplied_dti':m['supplied_dti_percent'],'corrected_dti':m['dti_percent'],
        'postclosing_reserve_months':m['reserve_months'],'cash_shortfall':m['cash_to_close_shortfall'],
        'original_expected':case['expected_decision']}))
print('Interpretation: corrected DTI exceeds the supplied policy ceiling of 50% in every case.')
print('Do not change fixtures or force model outputs to match inconsistent expected labels.')
''')
md('''## Part 2 Workflow and Orchestration
### 2A Specialist agents and coordination
The complete implementation below defines credit, income, asset, collateral, critic and decision agents. Each analysis receives sanitized application data, deterministic metrics and retrieved policy evidence. Prompts are triple-quoted and visible in code. All six analyses are displayed separately per supplied case in Part 3.

### 2B State and workflow graph
A typed state holds analyses, evidence, policy conditions, scores and timestamped audit events. A supervisor runs specialists sequentially, then critic and decision. Hard policy failures override score bands. Human review is required for risk >30, DTI >43%, LTV >90%, any bias flags or denial. The graph genuinely interrupts; the explicit simulation record resumes it. Raw PII never enters checkpoints.
''')
workflow=(REPO/'src/mortgage_engine/workflow.py').read_text(encoding='utf-8')
workflow=re.sub(r'^from \.core import .*\n|^from \.services import .*\n','',workflow,flags=re.M)
code(workflow)
code('''live_model = LiveModel(BASE_URL,API_KEY)
graph = build_workflow(live_model,policy_store)
print(graph.get_graph().draw_mermaid())
print('Graph compiled with in-memory checkpointing and real human-review interrupts.')
''')
md('''## Part 3 Testing and Analysis
### 3A Run all supplied test cases
All real model analyses and retrieved policy page references are retained. Human-review resumes below use clearly labeled **simulated** reviewer decisions for reproducible execution. They do not represent a person approving a real loan. Expected-label mismatches are reported honestly.
''')
code('''from langgraph.types import Command
results=[]
for index,case in enumerate(test_cases,1):
    config={'configurable':{'thread_id':f'submission-{index}'}}
    state=graph.invoke(initial_state(case),config)
    print('\\n'+'='*80+'\\nCASE '+case['case_id'])
    print('METRICS',json.dumps(state.get('metrics',{}),indent=2))
    for role,analysis in state.get('analyses',{}).items():
        print('\\nAGENT:',role.upper());print(analysis)
        print('RETRIEVED SOURCES:',[(e['source'],e['page']) for e in state['evidence'][role]])
    if state.get('status')=='ERROR': raise RuntimeError(state['error'])
    if state.get('__interrupt__'):
        print('REAL WORKFLOW INTERRUPT:',state['review_reasons'])
        review={'reviewer':'SIMULATED_COURSE_REVIEWER','decision':state['recommendation'],
                'notes':'Simulated demonstration: confirm unresolved policy findings; no actual human approval is claimed.', 'simulated':True}
        print('SIMULATED REVIEW RECORD:',json.dumps(review))
        state=graph.invoke(Command(resume=review),config)
    if state.get('status')=='ERROR':raise RuntimeError(state['error'])
    print('FINAL RECOMMENDATION:',state['recommendation'],'FINAL DEMO OUTCOME:',state['final_decision'])
    print('RISK BREAKDOWN:',state['risk_breakdown'],'TOTAL:',state['risk_score'])
    print('POLICY VIOLATIONS:',state['policy_violations']);print('CONDITIONS:',state['conditions'])
    print('BIAS SIGNALS:',state['bias_flags'])
    print('CREDIT MEMO:',state['decision_memo'])
    print('EXECUTIVE SUMMARY:',state['executive_summary'])
    print('AUDIT:',json.dumps(state['audit'],indent=2))
    results.append(state)
print('Verified live LLM responses:',live_model.calls)
''')
md('### 3B Evaluation and interpretation\nExpected decisions are normalized for comparison, without modifying fixtures. Differences are explained using calculated facts, not labeled as model success.')
code('''normalization={'APPROVED':'APPROVED','CONDITIONAL':'CONDITIONAL_APPROVAL','REJECTED':'DENIED'}
for case,result in zip(test_cases,results):
    expected=normalization[case['expected_decision']]
    print(case['case_id'],'Expected:',expected,'Observed:',result['final_decision'],
          'Match:',expected==result['final_decision'],'Human review:',result['human_review_completed'])
    assert result['metrics']['dti_percent']>50
    assert result['final_decision']=='DENIED'
    assert result['human_review_completed']
print('Observed outcomes are consistent with corrected policy calculations, but first two differ from supplied expected labels.')
print('Reasoning follows calculations -> evidence -> specialist review -> critic -> policy gates -> simulated human review.')
print('Critical factors: DTI in all cases; case 3 additionally fails credit minimum and cash to close, with serious repair concerns.')
''')
md('### 3C Error handling and additional regression tests\nThese tests explicitly use test doubles to verify failures, neutral cases, genuine interrupt/resume, rejected overrides and threshold boundaries without extra API costs. They are separate from the live demonstrations above.')
tests=(REPO/'tests/test_engine.py').read_text(encoding='utf-8')
tests=re.sub(r'^from mortgage_engine[^\n]+\n','',tests,flags=re.M)
code(tests)
code('''import inspect
regression_count=0
for test_name,func in list(globals().items()):
    if test_name.startswith('test_') and callable(func):
        if test_name=='test_invalid_income':
            for value in [0,-1,float('nan'),float('inf'),True]:func(value);regression_count+=1
        else:func();regression_count+=1
        print('PASS',test_name)
print('Regression assertions passed:',regression_count)
assert len(results)==3 and all(r['status']=='COMPLETE' for r in results)
''')
md('''## Part 4 Summary and Future Scope
### 4A Summary and observations
Deterministic tools revealed that the supplied DTI fields omit proposed housing payments and that the starter also counts aggregate debts twice. Correcting these defects changes the financial interpretation: all three supplied cases exceed the educational policy ceiling, so the first two expected outcomes cannot be treated as ground truth. The third case additionally has insufficient closing funds, a subminimum conventional score and repair concerns.

The system demonstrates live policy embeddings and Chroma retrieval, six separate LLM roles, supervisor routing, critic review, policy-controlled recommendations, actual interrupt/resume, simulated review records, credit memos and executive summaries. Retrieved policy excerpts and page references make the source evidence inspectable. Review triggers remain independent of the model narrative. Separate synthetic regression cases demonstrate low-risk completion and failure handling that the three flawed supplied fixtures cannot cover.

The execution outputs above are the evidence for observed behavior. We do not claim real-world accuracy, processing-time savings, legal compliance or fairness from these three examples.

### 4B Future scope
1. Expand the labeled evaluation set with reviewed financial inputs, realistic counterfactual pairs, protected-class fairness analysis and expert-adjudicated outcomes.
2. Add program-specific rules, structured verification of income and property documents, citation validation, bounded critic re-analysis and calibrated scoring based on representative historical data.
3. Implement authenticated reviewers, encrypted durable checkpoints, access-controlled audit retention, policy versioning, deployment monitoring and approved integrations with real lending systems.

### Limitations
The assignment policy PDF is an educational source, not a statement of current law. Loan program is missing from fixtures; conventional rules are assumed. Descriptions of evidence do not establish actual document verification. Bias screening is a keyword heuristic with false positives and false negatives. Risk scores are transparent course heuristics, not probabilities of default. Specialized tools are invoked deterministically before model interpretation, rather than selected autonomously by the LLM. Checkpoints are process-local. All final human-review records are explicitly simulated. No live credit bureau, notifications, LOS integration or production deployment is claimed.

### References
- Private supplied underwriting policy PDF, with retrieved page citations shown in Part 3.
- LangGraph interrupt/resume reference: https://reference.langchain.com/python/langgraph/types/interrupt
- Chroma Python reference: https://docs.trychroma.com/reference/python
''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':sys.version.split()[0]}})
path=OUT/'Senior_Mortgage_Underwriting_Submission.ipynb'
nbf.write(nb,path)
if '--prepare-only' in sys.argv:
    print('Prepared self-contained notebook; not executed and not submission-ready.')
    raise SystemExit(0)
# Launch a fresh kernel from this exact interpreter, with local credential path only in its environment.
os.environ['MORTGAGE_ENV_FILE']=str(REPO/'.env')
from jupyter_client import KernelManager
km=KernelManager(kernel_name='python3')
km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client=NotebookClient(nb,timeout=900,km=km,resources={'metadata':{'path':str(OUT)}},allow_errors=False)
try:
    client.on_cell_complete = lambda cell, cell_index, **kwargs: print('Executed cell', cell_index + 1, flush=True)
    client.execute()
except Exception:
    nbf.write(nb,path)
    print('Notebook stopped; saved actual outputs for diagnosis. No completion claim.')
    raise
nbf.write(nb,path)
exporter=HTMLExporter(template_name='lab')
body,_=exporter.from_notebook_node(nb)
htmlpath=path.with_suffix('.html');htmlpath.write_text(body,encoding='utf-8')
# Exact configured credential and conservative common credential patterns.
secret=dotenv_values(REPO/'.env').get('OPENAI_API_KEY','')
for artifact in (path,htmlpath):
    text=artifact.read_text(encoding='utf-8')
    if secret and secret in text:raise RuntimeError('Secret scan failed')
    if re.search(r'\b(?:sk-[A-Za-z0-9_-]{20,}|gl-[A-Za-z0-9+/=]{30,})',text):raise RuntimeError('Credential-pattern scan failed')
codecells=[c for c in nb.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in codecells)
assert not any(o.output_type=='error' for c in codecells for o in c.outputs)
report='''# Submission readiness report

- Rubric coverage: all 20 marks mapped in notebook opening table and corresponding headings.
- Execution: fresh kernel, all code cells executed sequentially, no unhandled errors.
- Code and outputs: full implementation embedded; visible live analyses, retrieval, metrics, review records, audit and regression results.
- HTML: exported directly from the final executed notebook.
- Secrets: exact configured credential and common credential patterns absent from both artifacts.
- Original fixtures: preserved; corrected DTI and expected-outcome discrepancies documented.
- Review: interrupt/resume demonstrated with explicitly simulated reviewer records.
- Publication: submission artifacts are outside repository; *.ipynb, *.html and *.pdf excluded by repository ignore rules.
- Limitations: documented in Part 4; no legal certification or actual human review claimed.

'''
for artifact in (path,htmlpath):report+=str(artifact)+'\nSHA256 '+hashlib.sha256(artifact.read_bytes()).hexdigest()+'\n\n'
(OUT/'submission-readiness.md').write_text(report,encoding='utf-8')
print('PASS: executed notebook, matching HTML and readiness report created. Code cells:',len(codecells))
