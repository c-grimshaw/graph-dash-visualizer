from datetime import datetime
import json
import logging
from pathlib import Path

import dash
import dash_cytoscape as cyto
import dash_mantine_components as dmc
import networkx as nx
from dash import Input, Output, State, callback, html, dcc
from dash_iconify import DashIconify
from grandcypher import GrandCypher

# Constants
LAYOUT_OPTIONS = [
    {"value": "cose", "label": "Force-directed"},
    {"value": "circle", "label": "Circle"},
    {"value": "grid", "label": "Grid"},
    {"value": "random", "label": "Random"},
    {"value": "breadthfirst", "label": "Hierarchical"},
]

NODE_STYLES = {
    "person": "#e74c3c",
    "project": "#f39c12", 
    "technology": "#9b59b6",
    "organization": "#2ecc71",
}

DEFAULT_QUERY = "MATCH (n) RETURN n"

app = None
graph_db = None
SAMPLE_QUERIES = []

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_app():
    """Initialize the application with logging, data, and layout."""
    logger.info("Initializing Graph Dash Visualizer")
    
    # Initialize Dash app
    app = dash.Dash(
        __name__,
        suppress_callback_exceptions=True,
    )
    
    # Load sample queries
    data_dir = Path(__file__).parent / "data"
    with open(data_dir / "sample_queries.txt") as f:
        sample_queries = [line.strip() for line in f if line.strip()]
    
    # Initialize graph database
    graph_db = GraphDatabase()
    
    # Set app layout
    app.layout = create_app_shell(sample_queries)
    
    logger.info("Application initialization complete")
    return app, graph_db, sample_queries


class GraphDatabase:
    """
    Handles graph data using NetworkX and executes Cypher queries with Grand-Cypher.
    Provides a robust backend for graph operations and visualization.
    """

    def __init__(self):
        logging.info("Initializing GraphDatabase with NetworkX")
        self.graph = nx.MultiDiGraph()
        self.initialize_sample_data()
        logging.info(f"GraphDatabase initialized with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")

    def initialize_sample_data(self):
        """Initializes the database with a sample graph dataset using NetworkX."""
        self.graph.clear()
        
        data_dir = Path(__file__).parent / "data"
        
        with open(data_dir / "sample_nodes.json") as f:
            nodes_data = json.load(f)
        
        with open(data_dir / "sample_edges.json") as f:
            edges_data = json.load(f)
        
        for node in nodes_data:
            node_id = node.pop("id")
            self.graph.add_node(node_id, **node)
        
        for edge in edges_data:
            label = edge.pop("label")
            self.graph.add_edge(edge["source"], edge["target"], __labels__={label}, **edge)
        
        logging.info(f"Sample data loaded: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

    def execute_cypher(self, query):
        """Executes a Cypher query using Grand-Cypher on the NetworkX graph."""
        try:
            query = query.strip()
            logging.info(f"Executing: {query}")
            
            if not query:
                return {"nodes": [], "edges": [], "error": None, "stats": {}, "query_result": None}

            start_time = datetime.now()

            if query.upper() == "DELETE ALL":
                self.initialize_sample_data()
                query_result = "Graph data reinitialized"
            else:
                grand_cypher = GrandCypher(self.graph)
                query_result = grand_cypher.run(query)

            graph_data = self._get_all_graph_data()
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            graph_data["stats"]["execution_time"] = execution_time
            graph_data["query_result"] = query_result
            
            return graph_data

        except Exception as e:
            logging.error(f"Query failed: {e}")
            return {
                "nodes": [], "edges": [], "error": str(e),
                "stats": {"execution_time": 0}, "query_result": f"Query failed: {str(e)}",
            }

    def _get_all_graph_data(self):
        """Returns the entire graph formatted for Cytoscape."""
        nodes = [self._node_to_cytoscape(node_id) for node_id in self.graph.nodes()]
        edges = [
            self._edge_to_cytoscape(source, target, edge_data)
            for source, target, edge_data in self.graph.edges(data=True)
        ]
        return {
            "nodes": nodes, "edges": edges, "error": None,
            "stats": {"node_count": len(nodes), "edge_count": len(edges)},
        }

    def _node_to_cytoscape(self, node_id):
        """Converts a NetworkX node to a Cytoscape-compatible dict."""
        node_data = self.graph.nodes[node_id]
        node_type = node_data.get("type", "Unknown")
        
        cyto_data = {k: str(v) for k, v in node_data.items()}
        cyto_data.update({
            "id": str(node_id),
            "label": str(node_data.get("name", node_id)),
            "type": str(node_type),
        })
        
        return {"data": cyto_data, "classes": str(node_type).lower()}

    def _edge_to_cytoscape(self, source, target, edge_data):
        """Converts a NetworkX edge to a Cytoscape-compatible dict."""
        edge_labels = edge_data.get("__labels__", {"relationship"})
        edge_label = list(edge_labels)[0] if edge_labels else "relationship"
        edge_id = f"{source}_{target}_{edge_label}"
        
        cyto_data = {k: str(v) for k, v in edge_data.items()}
        cyto_data.update({
            "id": str(edge_id), "source": str(source), "target": str(target), "label": str(edge_label),
        })
        
        return {"data": cyto_data}


# App shell layout
def create_app_shell(sample_queries):
    return dmc.MantineProvider(
        children=[
            dmc.AppShell(
                padding="md",
                header={"height": 60},
                children=[
                    dmc.AppShellHeader(
                        children=[
                            dmc.Title("Graph Dash", order=3, px="md"),
                        ],
                    ),
                    dmc.AppShellMain(children=create_graph_visualizer(sample_queries)),
                ],
            ),
        ],
    )


def create_sidebar_panel(title, children):
    """Create a standardized sidebar panel."""
    return dmc.Paper(
        p="md",
        withBorder=True,
        children=dmc.Stack(gap="sm", children=[dmc.Text(title, fw="bold")] + children),
    )


def create_graph_visualizer(sample_queries):
    return dmc.Container(
        fluid=True,
        mt="xl",
        pt="md",
        children=[
            dmc.Paper(
                p="md",
                mb="md",
                withBorder=True,
                children=[
                    dmc.Stack(
                        gap="md",
                        children=[
                            dmc.Group(
                                justify="space-between",
                                children=[
                                    dmc.Title("Graph Visualizer", order=2),
                                    dmc.Group(
                                        children=[
                                            dmc.Select(
                                                placeholder="Sample queries...",
                                                data=sample_queries,
                                                w=300,
                                                id="sample-queries",
                                                clearable=True,
                                            ),
                                            dmc.Button(
                                                "Clear Graph",
                                                variant="outline",
                                                color="red",
                                                id="clear-graph-btn",
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            dmc.Textarea(
                                placeholder="Enter your Cypher query...",
                                minRows=4,
                                id="cypher-input",
                                value=DEFAULT_QUERY,
                            ),
                            dmc.Group(
                                children=[
                                    dmc.Button(
                                        "Execute Query",
                                        leftSection=DashIconify(
                                            icon="radix-icons:play", width=16
                                        ),
                                        id="execute-btn",
                                    ),
                                    dmc.Button(
                                        "Clear Query", variant="outline", id="clear-btn"
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            dmc.Grid(
                children=[
                    dmc.GridCol(
                        span=9,
                        children=dmc.Paper(
                            p="md",
                            withBorder=True,
                            h=600,
                            children=cyto.Cytoscape(
                                id="cytoscape-graph",
                                elements=[],
                                layout={"name": "cose", "animate": True},
                                style={"width": "100%", "height": "550px"},
                                stylesheet=get_cytoscape_stylesheet(),
                            ),
                        ),
                    ),
                    dmc.GridCol(
                        span=3,
                        children=dmc.Stack(
                            gap="md",
                            children=[
                                create_sidebar_panel("Layout", [
                                    dmc.Select(data=LAYOUT_OPTIONS, value="cose", id="layout-select"),
                                ]),
                                create_sidebar_panel("Query Results", [
                                    html.Div(id="query-stats"),
                                ]),
                                create_sidebar_panel("Result", [
                                    dmc.Group(
                                        justify="space-between",
                                        align="center",
                                        children=[
                                            dmc.Text("Result", fw="bold"),
                                            dcc.Clipboard(
                                                id="clipboard",
                                                style={
                                                    "display": "inline-block",
                                                    "cursor": "pointer",
                                                    "transition": "opacity 0.2s",
                                                },
                                                className="clipboard-button",
                                            ),
                                        ],
                                    ),
                                    dmc.ScrollArea(
                                        h=200,
                                        children=html.Pre(
                                            id="query-results",
                                            style={
                                                "fontSize": "12px",
                                                "whiteSpace": "pre-wrap",
                                                "wordBreak": "break-word",
                                            }
                                        )
                                    ),
                                ]),
                                create_sidebar_panel("Selected Element", [
                                    html.Div(id="selected-info"),
                                ]),
                            ],
                        ),
                    ),
                ],
            ),
            html.Div(id="error-display"),
        ],
    )


def get_cytoscape_stylesheet():
    base_node_style = {
        "content": "data(label)",
        "width": "60px",
        "height": "60px",
        "font-size": "10px",
        "text-valign": "center",
        "text-halign": "center",
        "text-wrap": "wrap",
        "text-max-width": "50px",
        "background-color": "#3498db",
        "color": "white",
    }
    
    styles = [{"selector": "node", "style": base_node_style}]
    
    # Add node type styles
    for node_type, color in NODE_STYLES.items():
        styles.append({"selector": f".{node_type}", "style": {"background-color": color}})
    
    # Add edge style
    styles.append({
        "selector": "edge",
        "style": {
            "content": "data(label)",
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "line-color": "#95a5a6",
            "target-arrow-color": "#95a5a6",
            "font-size": "10px",
        },
    })
    
    return styles


def _build_cypher_response(result):
    elements = result.get("nodes", []) + result.get("edges", [])
    stats = result.get("stats", {})

    stats_display = [
        dmc.Text(f"Nodes: {stats.get('node_count', 0)}", size="sm"),
        dmc.Text(f"Edges: {stats.get('edge_count', 0)}", size="sm"),
        dmc.Text(f"Execution time: {stats.get('execution_time', 0):.2f}ms", size="sm"),
    ]

    error_display = []
    if result.get("error"):
        error_display = [
            dmc.Alert(
                title="Query Error",
                children=result["error"],
                color="red",
                icon=DashIconify(icon="radix-icons:exclamation-triangle"),
            )
        ]

    return elements, stats_display, error_display


@callback(
    Output("cytoscape-graph", "elements"),
    Output("query-stats", "children"),
    Output("error-display", "children"),
    Output("query-results", "children"),
    Input("execute-btn", "n_clicks"),
    Input("clear-graph-btn", "n_clicks"),
    State("cypher-input", "value"),
)
def update_graph(execute_clicks, clear_clicks, query):
    """Main callback to update the graph based on user actions."""
    try:
        # Determine action based on which button was clicked
        if clear_clicks:
            # Clear graph button was clicked
            graph_db.initialize_sample_data()
            result = graph_db.execute_cypher(DEFAULT_QUERY)
        elif execute_clicks and query:
            # Execute button was clicked with a query
            result = graph_db.execute_cypher(query)
        else:
            # Initial load or no action
            result = graph_db.execute_cypher(DEFAULT_QUERY)
        
        elements, stats_display, error_display = _build_cypher_response(result)
        query_results_text = _format_query_result(result.get("query_result"))
        
        return elements, stats_display, error_display, query_results_text
    
    except Exception as e:
        logging.error(f"Callback error: {e}")
        return _handle_callback_error(e)


def _format_query_result(query_result):
    """Format query result for display."""
    if query_result is None:
        return "No query executed"
    
    if isinstance(query_result, (dict, list)):
        try:
            return json.dumps(query_result, indent=2, default=str)
        except (TypeError, ValueError):
            return str(query_result)
    return str(query_result)


def _handle_callback_error(error):
    """Handle callback errors with fallback data."""
    try:
        fallback_result = graph_db._get_all_graph_data()
        elements = fallback_result.get("nodes", []) + fallback_result.get("edges", [])
    except Exception:
        elements = []
    
    error_display = [
        dmc.Alert(
            title="Application Error",
            children=f"An error occurred: {str(error)}",
            color="red",
            icon=DashIconify(icon="radix-icons:exclamation-triangle"),
        )
    ]
    
    stats_display = [dmc.Text("Error occurred", size="sm", c="red")]
    query_results_text = f"Error: {str(error)}"
    
    return elements, stats_display, error_display, query_results_text


@callback(
    Output("cypher-input", "value"),
    Input("sample-queries", "value"),
    Input("clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def update_query_input(sample_query, clear_clicks):
    """Update query input based on user actions."""
    # Determine action based on which input changed
    if clear_clicks:
        # Clear button was clicked
        return ""
    elif sample_query:
        # Sample query was selected
        return sample_query
    
    return dash.no_update


@callback(Output("cytoscape-graph", "layout"), Input("layout-select", "value"))
def update_layout(layout_name):
    return {"name": layout_name, "animate": True} if layout_name else dash.no_update


@callback(
    Output("selected-info", "children"),
    Input("cytoscape-graph", "selectedNodeData"),
    Input("cytoscape-graph", "selectedEdgeData"),
)
def display_selected_element(node_data, edge_data):
    if not node_data and not edge_data:
        return dmc.Text("Select a node or edge to view details", size="sm", c="gray")

    data = node_data[0] if node_data else edge_data[0]
    element_type = "Node" if node_data else "Edge"
    
    return _build_element_details(data, element_type, node_data)


def _build_element_details(data, element_type, is_node):
    """Build detailed element information display."""
    title = f"Type: {element_type}"
    if is_node:
        title += f" ({data.get('type', 'Unknown')})"

    core_properties = (
        {"id", "label", "type"} if is_node else {"id", "source", "target", "label"}
    )

    details = [dmc.Text(title, size="sm", fw="bold")]
    
    # Add core properties
    if is_node:
        details.extend([
            dmc.Text(f"ID: {data.get('id', 'N/A')}", size="sm"),
            dmc.Text(f"Label: {data.get('label', 'N/A')}", size="sm"),
        ])
    else:
        details.extend([
            dmc.Text(f"Source: {data.get('source', 'N/A')}", size="sm"),
            dmc.Text(f"Target: {data.get('target', 'N/A')}", size="sm"),
            dmc.Text(f"Label: {data.get('label', 'N/A')}", size="sm"),
        ])

    # Add additional properties
    details.append(html.Hr())
    details.append(dmc.Text("Properties:", size="sm", fw="bold"))

    properties = {k: v for k, v in data.items() if k not in core_properties}
    if properties:
        details.extend([dmc.Text(f"{k}: {v}", size="xs") for k, v in properties.items()])
    else:
        details.append(dmc.Text("No additional properties.", size="xs", c="gray"))

    return details


@callback(
    Output("clipboard", "content"),
    Input("query-results", "children"),
)
def update_clipboard_content(query_results):
    return query_results or "No query executed"


if __name__ == "__main__":
    app, graph_db, SAMPLE_QUERIES = init_app()
    app.run(debug=True, host="0.0.0.0", port=8050)
