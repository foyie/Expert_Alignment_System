import datetime
from pathlib import Path

def generate_report(metrics: dict, llm_scores: list, output_dir: Path = Path("results")) -> str:
    """Generate markdown evaluation report"""
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md"
    
    report_content = f"""
    # Evaluation Report
    
    ## Summary Metrics
    - Precision@{metrics.get('k', 3)}: {metrics['precision@k']:.2f}
    - Mean Reciprocal Rank: {metrics['mrr']:.2f}
    - NDCG@{metrics.get('k', 3)}: {metrics['ndcg']:.2f}
    - Cases Evaluated: {metrics['num_cases']}
    
    ## LLM Evaluation Samples
    {generate_llm_section(llm_scores)}
    
    ## Recommendations
    {generate_recommendations(metrics)}
    """
    
    with open(report_path, 'w') as f:
        f.write(report_content.strip())
    
    return report_path

def generate_llm_section(scores: list) -> str:
    return "\n".join(
        f"### Match {i+1}\n- Score: {s['score']}\n- Reason: {s['reason']}\n"
        for i, s in enumerate(scores[:3])  # Show top 3 samples
    )

def generate_recommendations(metrics: dict) -> str:
    recs = []
    if metrics['precision@k'] < 0.7:
        recs.append("Consider enhancing skill extraction from project descriptions")
    if metrics['mrr'] < 0.5:
        recs.append("Improve ranking algorithm for critical first positions")
    return "\n".join(f"- {r}" for r in recs)