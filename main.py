from src.matching_system import ProjectMatcher
from src.utils import load_briefs, load_experts

if __name__ == "__main__":
    # Initialize system
    matcher = ProjectMatcher()
    
    # Load sample data
    briefs = load_briefs("data/briefs.json")
    experts = load_experts("data/experts.json")
    
    # Example usage
    project = briefs[0]
    matches = matcher.find_matches(project)
    print("Top matches:", [m["name"] for m in matches])