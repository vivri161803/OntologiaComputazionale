import argparse
import os
import sys

# Import the classes from the current module
from graph_analyzer import GraphAnalyzer
from dashboard_visualizer import DashboardVisualizer

def main():
    parser = argparse.ArgumentParser(description="Create a comparative dashboard for KG statistics.")
    parser.add_argument(
        '--inputs', 
        nargs='+', 
        required=True, 
        help='List of paths to .tsv files containing the knowledge graphs.'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='kg_dashboard.png', 
        help='Path to save the generated dashboard image (e.g. kg_dashboard.png)'
    )
    
    args = parser.parse_args()

    data_list = []

    for filepath in args.inputs:
        print(f"[*] Processing {filepath}...")
        if not os.path.exists(filepath):
            print(f"[!] Warning: File {filepath} not found. Skipping.")
            continue
        
        # Extract book/graph name from filepath
        base_name = os.path.basename(filepath)
        graph_name, _ = os.path.splitext(base_name)

        analyzer = GraphAnalyzer(filepath)
        try:
            G = analyzer.load_graph()
            print(f"    - Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            
            stats = analyzer.compute_statistics()
            data_list.append((graph_name, G, stats))
            
            print(f"    - Statistics computed successfully.")
        except Exception as e:
            print(f"[!] Error processing {filepath}: {e}")

    if not data_list:
        print("No valid graphs to display. Exiting.")
        sys.exit(1)

    print("\n[*] Generating comparative dashboard...")
    visualizer = DashboardVisualizer(data_list)
    visualizer.plot_dashboard(save_path=args.output)
    print("[*] DONE!")

if __name__ == '__main__':
    main()
