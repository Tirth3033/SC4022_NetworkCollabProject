import os
import math
import random
import requests
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt

def load_input_data(input_path):
    """Load and clean the input Excel file of data scientists.
    Returns a list of dicts for each scientist with cleaned information."""
    import pandas as pd

    # Read Excel (supports .xls or .xlsx). Use appropriate engine if needed.
    try:
        df = pd.read_excel(input_path)
    except Exception as e:
        # If .xls, try xlrd engine (ensure xlrd installed for .xls files).
        try:
            df = pd.read_excel(input_path, engine='xlrd')
        except Exception as e2:
            raise RuntimeError(f"Failed to read input file {input_path}: {e2}")
    # Standardize column names for easier access
    df.columns = [str(col).strip().lower().replace(' ', '_').replace('/', '_') for col in df.columns]
    # Identify the column that contains the DBLP URL/ID
    dblp_col = None
    for col in df.columns:
        if 'dblp' in col:  # e.g., 'dblp_id_url' or 'dblp' etc.
            dblp_col = col
            break
    if dblp_col is None:
        raise KeyError("No DBLP URL/ID column found in input file.")
    # Remove duplicate records based on the DBLP identifier (keep last occurrence)
    df = df.drop_duplicates(subset=dblp_col, keep='last').reset_index(drop=True)
    # Assign random expertise 1-10 for missing or invalid expertise values
    if 'expertise' in df.columns:
        df['expertise'] = df['expertise'].apply(lambda x: int(x) if pd.notna(x) and str(x).isdigit() else random.randint(1, 10))
    else:
        df['expertise'] = [random.randint(1, 10) for _ in range(len(df))]
    # Ensure essential fields exist
    if 'name' not in df.columns:
        df['name'] = None
    if 'country' not in df.columns:
        df['country'] = None
    if 'institution' not in df.columns:
        df['institution'] = None
    records = df.to_dict(orient='records')
    return records, dblp_col

def resolve_dblp_pid(url):
    try:
        r = requests.get(url, allow_redirects=True, timeout=10)
        pid = '/'.join(r.url.split('/')[-2:]).replace('.html','')
        return pid, r.url
    except Exception as e:
        print(f"❌ Failed to access link: {url} — {e}")
        return None, None

def fetch_dblp_author_xml(pid):
    """Fetch the DBLP XML for an author given their PID.
    Returns a BeautifulSoup parsed XML, or None on failure."""
    if pid is None:
        return None
    url = f"https://dblp.org/pid/{pid}.xml"
    try:
        xml_data = requests.get(url).content
        try:
            soup = BeautifulSoup(xml_data, "lxml-xml")  # Explicit XML mode with lxml
        except Exception:
            soup = BeautifulSoup(xml_data, "xml")  # fallback
        return soup
    except Exception as e:
        print(f"❌ Failed to fetch or parse XML for PID {pid}: {e}")
        return None


def build_collaboration_graph(records, dblp_col):
    """Build the co-authorship network (NetworkX graph) from the list of scientist records.
    Returns the graph and a dict of edges with their first collaboration year."""
    import time  # needed for sleep
    G = nx.Graph()
    # Dictionary to map official DBLP names to PID and record, for matching coauthors
    name_to_pid = {}
    # Dictionary to store edges and earliest year of collaboration
    edge_first_year = {}

    print(f"📄 Total scientists to process: {len(records)}")

    # First, add all nodes (scientists) to the graph with their attributes
    for idx, rec in enumerate(records):
        # We'll fill in the official name after fetching DBLP data
        pid = rec.get(dblp_col) or rec.get('dblp') or rec.get('dblp_id_url') or rec.get('dblp_url')
        # Normalize PID by resolving URL if needed
        pid_val, pid_url = resolve_dblp_pid(pid)
        if pid_val is None:
            # If cannot resolve, use the original value (but then fetching XML may fail)
            pid_val = str(pid) if pid is not None else None
        rec['pid'] = pid_val
        # Add node with a provisional name (will update if we get official name from XML)
        node_id = pid_val if pid_val is not None else f"Unknown_{len(G)}"
        # Use PID as unique node identifier in the graph
        G.add_node(node_id, name=rec.get('name'), country=rec.get('country'),
                   institution=rec.get('institution'), expertise=rec.get('expertise'))
        # Map input name to PID initially (fallback for matching if needed)
        if rec.get('name'):
            name_to_pid[rec['name']] = node_id

        print(f"➕ Added node {idx + 1}/{len(records)}: {rec.get('name')} ({node_id})")

    # Next, fetch and parse each author's DBLP XML to get coauthor relationships
    for idx, rec in enumerate(records):
        pid = rec.get('pid')
        if pid is None:
            print(f"⚠️ Skipping record {idx + 1}: No PID resolved.")
            continue  # skip if no PID resolved

        print(f"\n🔍 Processing DBLP XML for PID {pid} ({idx + 1}/{len(records)})")
        soup = fetch_dblp_author_xml(pid)
        if soup is None:
            print(f"❌ Failed to fetch or parse XML for PID {pid}")
            continue  # skip if XML could not be fetched

        # If we didn't update the official name yet, try to extract it from the XML.
        # The <dblpperson> tag may have name or the first <author> tag might be the primary name.
        # As a heuristic, use the input name if official name not directly available.
        dblp_person_tag = soup.find('dblpperson')
        official_name = None
        if dblp_person_tag and dblp_person_tag.has_attr('name'):
            official_name = dblp_person_tag['name']
        else:
            # Try an <author> tag that matches this person (in their own publications list).
            # Often the person's primary name appears as the first author in their publications or is listed separately.
            # As a heuristic, use the input name if official name not directly available.
            official_name = rec.get('name')
        if official_name:
            # Update graph node's name attribute to the official name (for consistency in coauthor matching)
            if G.has_node(pid):
                G.nodes[pid]['name'] = official_name
            # Add to mapping for coauthor name matching
            name_to_pid[official_name] = pid
            print(f"✅ Official name resolved: {official_name}")
        else:
            print(f"⚠️ No official name found for PID {pid}")

        # Find all publication entries (article, inproceedings, etc.) in the XML
        publication_tags = soup.find_all(['article', 'inproceedings', 'incollection', 'book', 'phdthesis', 'mastersthesis'])
        # (We skip 'proceedings' or 'editor' entries, focusing only on research papers/articles co-authored)
        print(f"📚 Found {len(publication_tags)} publications for {official_name or pid}")

        for pub in publication_tags:
            # Get list of authors for this publication
            authors = [a_tag.text for a_tag in pub.find_all('author')]
            # If main person is not in authors list, skip (e.g., they might be editor of this record, not an author)
            if official_name and official_name not in authors:
                continue
            year_tag = pub.find('year')
            pub_year = None
            if year_tag:
                # Ensure numeric year if possible
                try:
                    pub_year = int(year_tag.text)
                except:
                    pub_year = None
            # Add edges for each co-author in this publication who is in our list
            for coauthor_name in authors:
                if official_name and coauthor_name == official_name:
                    continue  # skip self
                if coauthor_name in name_to_pid:
                    coauthor_pid = name_to_pid[coauthor_name]
                    main_pid = pid  # current scientist's PID
                    # Only add if the coauthor_pid is also in graph (should be, if in list)
                    if G.has_node(coauthor_pid) and G.has_node(main_pid):
                        G.add_edge(main_pid, coauthor_pid)
                        print(f"🔗 Added edge: {main_pid} ↔ {coauthor_pid}")
                        # Record or update earliest year for this edge (undirected, use frozenset for key)
                        if pub_year:
                            edge_key = frozenset([main_pid, coauthor_pid])
                            if edge_key not in edge_first_year or (pub_year is not None and pub_year < edge_first_year[edge_key]):
                                edge_first_year[edge_key] = pub_year

        time.sleep(1.2)  # 💤 Pause to avoid rate-limiting

    print(f"\n✅ Done. Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G, edge_first_year

def analyze_network_properties(G, label="Network"):
    """Compute and print key properties of graph G."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    degrees = [d for _, d in G.degree()]  # degree sequence
    avg_degree = (sum(degrees) / num_nodes) if num_nodes > 0 else 0
    max_degree = max(degrees) if degrees else 0
    isolates = list(nx.isolates(G))
    num_isolates = len(isolates)
    # Connected components
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    num_components = len(comps)
    largest_comp_size = len(comps[0]) if num_components > 0 else 0
    # Clustering coefficients
    avg_clust = nx.average_clustering(G)
    transitivity = nx.transitivity(G)  # global clustering coefficient
    # Top hubs (nodes with highest degree)
    top_hubs = sorted(G.degree, key=lambda x: x[1], reverse=True)[:5]

    print(f"--- {label} Properties ---")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Maximum degree: {max_degree}")
    print(f"Average clustering coefficient: {avg_clust:.4f}")
    print(f"Transitivity (global clustering): {transitivity:.4f}")
    print(f"Number of connected components: {num_components}")
    print(f"Size of largest component: {largest_comp_size}")
    print(f"Number of isolated nodes (degree 0): {num_isolates}")
    if num_isolates > 0:
        iso_percentage = (num_isolates/num_nodes)*100 if num_nodes>0 else 0
        print(f"  (Isolates are {iso_percentage:.1f}% of all nodes)")
    if top_hubs:
        hub_names = []
        for node, deg in top_hubs:
            name = G.nodes[node].get('name', node)
            hub_names.append(f"{name} (deg {deg})")
        print("Top hubs by degree: " + ", ".join(hub_names))
    print("---------------------------")

def plot_degree_distribution(G, filename, title="Degree Distribution", compare_graph=None, compare_label=""):
    """Plot and save the degree distribution histogram of graph G.
       If compare_graph is provided, overlay its degree distribution for comparison."""
    degrees = [d for _, d in G.degree()]
    max_deg = max(degrees) if degrees else 0
    bins = range(0, max_deg+2)  # bins from 0 to max_deg
    plt.figure(figsize=(6,4))
    plt.hist(degrees, bins=bins, alpha=0.7, color='steelblue', edgecolor='black', label=f"{title}")
    if compare_graph:
        comp_degrees = [d for _, d in compare_graph.degree()]
        # Use same bin range extended to cover compare graph's max degree if larger
        comp_max_deg = max(comp_degrees) if comp_degrees else 0
        if comp_max_deg > max_deg:
            bins = range(0, comp_max_deg+2)
        plt.hist(comp_degrees, bins=bins, alpha=0.5, color='orange', edgecolor='black', label=compare_label or "Comparison")
        max_deg = max(max_deg, comp_max_deg)
    plt.title(title if not compare_graph else f"{title} vs {compare_label}")
    plt.xlabel("Degree")
    plt.ylabel("Number of Nodes")
    plt.xticks(range(0, max_deg+1, max(1, max_deg//10)))  # tick every few degrees
    if compare_graph:
        plt.legend()
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()

def plot_metric_comparison(metrics, labels, filename, title="Network Metrics Comparison"):
    """Plot a simple bar chart comparing given metrics for original vs another network.
       `metrics` should be a list of tuples [(metric_name, value_orig, value_compare), ...],
       and `labels` is a tuple of (label_orig, label_compare)."""
    metric_names = [m[0] for m in metrics]
    orig_values = [m[1] for m in metrics]
    comp_values = [m[2] for m in metrics]
    x = range(len(metrics))
    plt.figure(figsize=(6,4))
    width = 0.35
    plt.bar([i - width/2 for i in x], orig_values, width, label=labels[0], color='steelblue')
    plt.bar([i + width/2 for i in x], comp_values, width, label=labels[1], color='sandybrown')
    plt.xticks(x, metric_names)
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    for i, val in enumerate(orig_values):
        plt.text(i - width/2, val + 0.05*max(orig_values+comp_values), f"{val}", ha='center', va='bottom', fontsize=8)
    for i, val in enumerate(comp_values):
        plt.text(i + width/2, val + 0.05*max(orig_values+comp_values), f"{val}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()

def analyze_temporal_evolution(G, edge_first_year):
    """Analyze and print how network properties evolve over time (yearly)."""
    if not edge_first_year:
        print("No edge data available for temporal evolution analysis.")
        return
    years = sorted({y for y in edge_first_year.values() if y is not None})
    if not years:
        print("No valid year information for edges.")
        return
    start_year, end_year = years[0], years[-1]
    print(f"\n--- Temporal Evolution of Collaboration Network ({start_year}–{end_year}) ---")
    cumulative_edges = []
    # Sort edges by year for cumulative addition
    edges_by_year = sorted(edge_first_year.items(), key=lambda kv: kv[1] if kv[1] is not None else float('inf'))
    # Keep track of current graph state (we will add edges year by year)
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))  # start with all nodes isolated
    current_index = 0
    for year in range(start_year, end_year+1):
        # add all edges that appear in this year
        while current_index < len(edges_by_year) and edges_by_year[current_index][1] is not None and edges_by_year[current_index][1] <= year:
            edge_key = edges_by_year[current_index][0]
            # edge_key is a frozenset of two nodes
            u, v = tuple(edge_key)
            H.add_edge(u, v)
            current_index += 1
        # Now compute metrics for this year
        edges_count = H.number_of_edges()
        comp_sizes = [len(c) for c in nx.connected_components(H)] if edges_count > 0 else [0]
        largest_comp = max(comp_sizes) if comp_sizes else 0
        avg_clust = nx.average_clustering(H) if edges_count > 0 else 0.0
        print(f"Year {year}: Edges={edges_count}, LargestComponent={largest_comp}, AvgClustering={avg_clust:.3f}")
    print("---------------------------------------------------------------\n")

def create_random_network_like(G):
    """Create a random network with same number of nodes and edges as graph G (G(n,m) model)."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    # Use NetworkX gnm_random_graph. This will create nodes labeled 0..n-1.
    rand_graph = nx.gnm_random_graph(n, m, seed=42)
    # Relabel nodes of random graph to match the labels of G (for easier comparison of attributes if needed).
    mapping = {old: new for old, new in zip(range(n), G.nodes())}
    rand_graph = nx.relabel_nodes(rand_graph, mapping)
    # (Now rand_graph has the same node identifiers as G, just randomly wired edges.)
    return rand_graph

def transform_network(G, kmax):
    """Transform the network by limiting degrees to at most kmax and increasing isolates, while preserving diversity."""
    G_trans = G.copy()
    # Sort nodes by degree descending (we will trim high-degree nodes first)
    high_degree_nodes = sorted([node for node in G_trans.nodes if G_trans.degree(node) > kmax],
                               key=lambda x: G_trans.degree(x), reverse=True)
    for node in high_degree_nodes:
        neighbors = list(G_trans.neighbors(node))
        if G_trans.degree(node) <= kmax:
            continue  # no trimming needed
        # Determine how many edges to remove
        excess = G_trans.degree(node) - kmax
        if excess <= 0:
            continue
        # If we must remove edges, choose which neighbors to KEEP (kmax of them) such that diversity is preserved.
        # We will select kmax neighbors that maximize diversity (covering different countries, institutions, expertise).
        # Get attributes of neighbors
        neighbor_attrs = []
        for nb in neighbors:
            attr = G_trans.nodes[nb]
            neighbor_attrs.append((nb, attr.get('country'), attr.get('institution'), attr.get('expertise')))
        selected = []  # neighbors to keep
        covered_values = {
            'country': set(),
            'institution': set(),
            'expertise': set()
        }
        # Greedy selection for diversity
        for _ in range(min(kmax, len(neighbor_attrs))):
            # Sort remaining neighbors by how many new attribute values they would contribute if selected
            best_nb = None
            best_new_cov = -1
            for (nb, country, inst, expert) in neighbor_attrs:
                if nb in selected:
                    continue
                new_cov = 0
                # Count new coverage for each attribute
                if country and country not in covered_values['country']:
                    new_cov += 1
                if inst and inst not in covered_values['institution']:
                    new_cov += 1
                if expert and expert not in covered_values['expertise']:
                    new_cov += 1
                # If this neighbor is currently connected to no one else (isolated if removed), perhaps prioritize? 
                # (Not strictly required, focusing on diversity.)
                # Choose the neighbor with the most new attribute coverage
                if new_cov > best_new_cov:
                    best_new_cov = new_cov
                    best_nb = (nb, country, inst, expert)
            if best_nb is None:
                # No neighbor provides new diversity; pick one with smallest degree (to possibly remove more hubs)
                neighbor_attrs.sort(key=lambda x: G_trans.degree(x[0]))
                best_nb = neighbor_attrs[0]
            # Select this neighbor
            nb, country, inst, expert = best_nb
            selected.append(nb)
            # Update covered attributes
            if country: covered_values['country'].add(country)
            if inst: covered_values['institution'].add(inst)
            if expert is not None: covered_values['expertise'].add(expert)
            # Remove the chosen neighbor from consideration list
            neighbor_attrs = [x for x in neighbor_attrs if x[0] != nb]
            if len(selected) >= kmax:
                break
        # Now `selected` contains up to kmax neighbors to keep connected; remove edges to all others
        for nb, country, inst, expert in neighbor_attrs:
            # Remove edge from node to this neighbor
            if G_trans.has_edge(node, nb):
                G_trans.remove_edge(node, nb)
    return G_trans

def compare_diversity(original_graph, transformed_graph):
    """Compute and print diversity (unique countries, institutions, expertise) in connected components of original vs transformed network."""
    # Get set of nodes that are in the giant component (or generally in any component >1) for each graph
    def get_connected_nodes(graph):
        # Nodes with degree > 0 (since isolates are not connected to anyone)
        return {n for n, deg in graph.degree() if deg > 0}
    orig_connected = get_connected_nodes(original_graph)
    trans_connected = get_connected_nodes(transformed_graph)
    # Gather unique values for each attribute among connected nodes
    def attribute_diversity(nodes, graph, attr):
        vals = set()
        for n in nodes:
            val = graph.nodes[n].get(attr)
            if val:
                vals.add(val)
        return vals
    orig_countries = attribute_diversity(orig_connected, original_graph, 'country')
    trans_countries = attribute_diversity(trans_connected, transformed_graph, 'country')
    orig_insts = attribute_diversity(orig_connected, original_graph, 'institution')
    trans_insts = attribute_diversity(trans_connected, transformed_graph, 'institution')
    orig_expertise = attribute_diversity(orig_connected, original_graph, 'expertise')
    trans_expertise = attribute_diversity(trans_connected, transformed_graph, 'expertise')
    print("\n--- Diversity in Connected Network: Original vs Transformed ---")
    print(f"Unique countries (connected component): {len(orig_countries)} -> {len(trans_countries)}")
    if orig_countries - trans_countries:
        lost = orig_countries - trans_countries
        print(f"   (Lost countries in transformed network: {', '.join(map(str, lost))})")
    print(f"Unique institutions (connected component): {len(orig_insts)} -> {len(trans_insts)}")
    if orig_insts - trans_insts:
        lost = orig_insts - trans_insts
        print(f"   (Lost institutions in transformed network: {', '.join(map(str, lost))})")
    print(f"Unique expertise values (connected component): {len(orig_expertise)} -> {len(trans_expertise)}")
    if orig_expertise - trans_expertise:
        lost = orig_expertise - trans_expertise
        print(f"   (Lost expertise values in transformed network: {sorted(lost)})")
    print("--------------------------------------------------------------\n")

if __name__ == "__main__":
    # Define input and output directories relative to current script location
    current_dir = os.getcwd()
    input_dir = os.path.join(current_dir, "Input")
    results_dir = os.path.join(current_dir, "Results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # Identify input files
    data_file = None
    config_file = None
    # Look for an Excel file (xls or xlsx) in Input directory
    if os.path.isdir(input_dir):
        for fname in os.listdir(input_dir):
            fpath = os.path.join(input_dir, fname)
            if fname.lower().endswith(('.xls', '.xlsx')):
                data_file = fpath
            elif fname.lower() == "config.txt":
                config_file = fpath
    if data_file is None:
        raise FileNotFoundError("Data scientists input file (.xls/.xlsx) not found in Input directory.")
    # Load and preprocess input data
    records, dblp_col = load_input_data(data_file)
    print(f"Loaded {len(records)} unique data scientist records from input.")
    # Build collaboration network graph
    G_real, edge_first_year = build_collaboration_graph(records, dblp_col)
    print(f"Constructed collaboration network with {G_real.number_of_nodes()} nodes and {G_real.number_of_edges()} edges.")
    # Analyze properties of the real network
    analyze_network_properties(G_real, label="Real Collaboration Network")
    # Plot degree distribution for real network
    plot_degree_distribution(G_real, filename="degree_distribution.png", title="Degree Distribution")
    # Temporal evolution analysis
    analyze_temporal_evolution(G_real, edge_first_year)
    # Create a random network for comparison (same size and edges)
    G_random = create_random_network_like(G_real)
    analyze_network_properties(G_random, label="Random Network")
    # Compare real vs random network via plots
    plot_degree_distribution(G_real, compare_graph=G_random, compare_label="Random Network",
                             filename="degree_compare_real_random.png", title="Degree Distribution")
    # We can also compare some key metrics in a bar chart
    real_comps = sorted(nx.connected_components(G_real), key=len, reverse=True)
    rand_comps = sorted(nx.connected_components(G_random), key=len, reverse=True)
    metrics_compare = [
        ("GiantCompSize", len(real_comps[0]) if real_comps else 0, len(rand_comps[0]) if rand_comps else 0),
        ("Components", nx.number_connected_components(G_real), nx.number_connected_components(G_random)),
        ("Isolates", nx.number_of_isolates(G_real), nx.number_of_isolates(G_random)),
        ("Clustering", round(nx.average_clustering(G_real), 4), round(nx.average_clustering(G_random), 4))
    ]
    plot_metric_comparison(metrics_compare, labels=("Real", "Random"), filename="real_vs_random_metrics.png",
                            title="Real vs Random Network Metrics")
    # Read kmax from config file (if provided)
    kmax = None
    if config_file:
        with open(config_file, 'r') as cfg:
            for line in cfg:
                # find first occurrence of a number in the file
                import re
                m = re.search(r'\d+', line)
                if m:
                    kmax = int(m.group())
                    break
    if kmax is None:
        # If not specified, choose a default (for example 5)
        kmax = 5
        print(f"Config file not found or kmax not specified. Using default kmax = {kmax}.")
    else:
        print(f"kmax (collaboration cutoff) = {kmax}")
    # Transform the network based on kmax
    G_trans = transform_network(G_real, kmax)
    print(f"Transformed network with kmax={kmax}: {G_trans.number_of_nodes()} nodes, {G_trans.number_of_edges()} edges.")
    analyze_network_properties(G_trans, label=f"Transformed Network (kmax={kmax})")
    # Compare degree distribution before vs after transformation
    plot_degree_distribution(G_real, compare_graph=G_trans, compare_label="Transformed",
                             filename="degree_compare_transformed.png", title="Degree Distribution")
    # Compare connectivity metrics (giant component size, isolates, components) for original vs transformed
    orig_comps = sorted(nx.connected_components(G_real), key=len, reverse=True)
    trans_comps = sorted(nx.connected_components(G_trans), key=len, reverse=True)
    connectivity_metrics = [
        ("GiantCompSize", len(orig_comps[0]) if orig_comps else 0, len(trans_comps[0]) if trans_comps else 0),
        ("Components", nx.number_connected_components(G_real), nx.number_connected_components(G_trans)),
        ("Isolates", nx.number_of_isolates(G_real), nx.number_of_isolates(G_trans))
    ]
    plot_metric_comparison(connectivity_metrics, labels=("Original", "Transformed"),
                            filename="connectivity_compare_transformed.png",
                            title=f"Original vs Transformed Connectivity (kmax={kmax})")
    # Analyze diversity preservation in the transformed network
    compare_diversity(G_real, G_trans)
