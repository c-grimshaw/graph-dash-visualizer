import json
import re
from datetime import datetime

import dash
import dash_cytoscape as cyto
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, callback, dcc, html
from dash_iconify import DashIconify
from py2neo import Node, Relationship

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;600;700;800;900&display=swap"
    ],
    suppress_callback_exceptions=True,
)


# Graph database class for handling Cypher queries with py2neo
class GraphDatabase:
    """
    Handles graph data and simulates Cypher query execution for the demo.
    Uses py2neo objects locally without connecting to a live database.
    """

    def __init__(self):
        self.local_graph = {}
        self.relationships = []
        self.initialize_sample_data()

    def initialize_sample_data(self):
        """Initializes the database with a sample graph dataset."""
        nodes = {
            "Person:Charlie": Node(
                "Person",
                name="Charlie",
                age=35,
                role="Developer",
                department="Engineering",
            ),
            "Person:Taylor": Node(
                "Person", name="Taylor", age=28, role="Designer", department="Design"
            ),
            "Person:Emilie": Node(
                "Person", name="Emilie", age=32, role="Manager", department="Operations"
            ),
            "Person:Chandra": Node(
                "Person",
                name="Chandra",
                age=40,
                role="Director",
                department="Leadership",
            ),
            "Person:Dhruv": Node(
                "Person", name="Dhruv", age=38, role="CTO", department="Leadership"
            ),
            "Organization:CANSOFCOM": Node(
                "Organization", name="CANSOFCOM", type="Military", sector="Defense"
            ),
            "Project:Alpha": Node(
                "Project", name="Project Alpha", status="Active", priority="High"
            ),
            "Project:Beta": Node(
                "Project", name="Project Beta", status="Planning", priority="Medium"
            ),
            "Technology:AI": Node(
                "Technology", name="AI Platform", type="Software", version="2.0"
            ),
            "Technology:Cloud": Node(
                "Technology",
                name="Cloud Infrastructure",
                type="Infrastructure",
                provider="AWS",
            ),
        }
        self.local_graph = nodes

        self.relationships = [
            Relationship(
                nodes["Person:Charlie"],
                "WORKS_WITH",
                nodes["Person:Taylor"],
                since="2023",
                project="Project Alpha",
            ),
            Relationship(
                nodes["Person:Charlie"],
                "COLLABORATES_WITH",
                nodes["Organization:CANSOFCOM"],
                since="2022",
                contract="Defense",
            ),
            Relationship(
                nodes["Person:Charlie"],
                "LEADS",
                nodes["Project:Alpha"],
                role="Tech Lead",
                since="2023",
            ),
            Relationship(
                nodes["Person:Charlie"],
                "DEVELOPS",
                nodes["Technology:AI"],
                expertise="Machine Learning",
            ),
            Relationship(
                nodes["Person:Taylor"],
                "WORKS_WITH",
                nodes["Person:Emilie"],
                since="2023",
                project="Project Beta",
            ),
            Relationship(
                nodes["Person:Taylor"],
                "REPORTS_TO",
                nodes["Person:Chandra"],
                since="2022",
            ),
            Relationship(
                nodes["Person:Taylor"],
                "REPORTS_TO",
                nodes["Person:Dhruv"],
                since="2022",
            ),
            Relationship(
                nodes["Person:Taylor"],
                "DESIGNS",
                nodes["Project:Beta"],
                role="Lead Designer",
            ),
            Relationship(
                nodes["Person:Emilie"],
                "REPORTS_TO",
                nodes["Person:Chandra"],
                since="2021",
            ),
            Relationship(
                nodes["Person:Emilie"],
                "REPORTS_TO",
                nodes["Person:Dhruv"],
                since="2021",
            ),
            Relationship(
                nodes["Person:Emilie"],
                "MANAGES",
                nodes["Project:Beta"],
                role="Project Manager",
            ),
            Relationship(
                nodes["Person:Emilie"],
                "OVERSES",
                nodes["Technology:Cloud"],
                responsibility="Infrastructure",
            ),
            Relationship(
                nodes["Person:Chandra"],
                "COLLABORATES_WITH",
                nodes["Person:Dhruv"],
                since="2020",
                level="Executive",
            ),
            Relationship(
                nodes["Person:Chandra"],
                "SPONSORS",
                nodes["Project:Alpha"],
                budget="High",
            ),
            Relationship(
                nodes["Person:Dhruv"],
                "SPONSORS",
                nodes["Project:Beta"],
                budget="Medium",
            ),
            Relationship(
                nodes["Person:Dhruv"],
                "APPROVES",
                nodes["Technology:AI"],
                decision="Strategic",
            ),
            Relationship(
                nodes["Organization:CANSOFCOM"],
                "FUNDS",
                nodes["Project:Alpha"],
                contract_value="High",
            ),
            Relationship(
                nodes["Organization:CANSOFCOM"],
                "USES",
                nodes["Technology:AI"],
                application="Defense",
            ),
            Relationship(
                nodes["Technology:AI"],
                "DEPENDS_ON",
                nodes["Technology:Cloud"],
                integration="API",
            ),
            Relationship(
                nodes["Technology:AI"],
                "IMPLEMENTS",
                nodes["Project:Alpha"],
                phase="Development",
            ),
            Relationship(
                nodes["Technology:Cloud"],
                "HOSTS",
                nodes["Project:Beta"],
                environment="Production",
            ),
        ]

    def execute_cypher(self, query):
        """Simulates the execution of a Cypher query."""
        try:
            query = query.strip().upper()
            if not query:
                return {"nodes": [], "edges": [], "error": None, "stats": {}}

            start_time = datetime.now()

            if query.startswith("MATCH"):
                result = self._execute_match_query(query)
            elif query.startswith("CREATE"):
                result = self._execute_create_query(query)
            elif query.startswith("DELETE"):
                self.local_graph.clear()
                self.relationships.clear()
                self.initialize_sample_data()
                result = self._get_all_graph_data()
            else:
                result = self._get_all_graph_data()

            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result["stats"]["execution_time"] = execution_time
            return result

        except Exception as e:
            return {
                "nodes": [],
                "edges": [],
                "error": str(e),
                "stats": {"execution_time": 0},
            }

    def _execute_match_query(self, query):
        """Simulates execution of a MATCH query."""

        # Define patterns for specific MATCH queries
        match_patterns = {
            "MATCH (N) RETURN N": lambda: self._get_all_graph_data(),
            "MATCH (P:PERSON) RETURN P": lambda: self._get_nodes_by_label("Person"),
            "CHARLIE": lambda: self._get_node_relationships("Person:Charlie"),
        }

        for pattern, handler in match_patterns.items():
            if pattern in query:
                return handler()

        # Handle queries for relationships by type
        rel_type_match = re.search(r"-\[.*?:(\w+)]->", query)
        if rel_type_match:
            rel_type = rel_type_match.group(1)
            return self._get_relationships_by_type(rel_type)

        return self._get_all_graph_data()

    def _execute_create_query(self, query):
        """Simulates execution of a CREATE query."""
        create_pattern = r"CREATE \((\w+):(\w+) \{([^}]+)\}\)"
        match = re.search(create_pattern, query, re.IGNORECASE)
        if match:
            _, node_label, properties_str = match.groups()
            props = dict(re.findall(r"(\w+):\s*'([^']*)'", properties_str))

            # Convert age to int if present
            if "age" in props:
                props["age"] = int(props["age"])

            node_id = f"{node_label.capitalize()}:{props.get('name', 'NewNode')}"
            self.local_graph[node_id] = Node(node_label.capitalize(), **props)

        return self._get_all_graph_data()

    def _get_nodes_by_label(self, label):
        """Gets all nodes with a specific label."""
        nodes = [
            self._node_to_cytoscape(node_id, node)
            for node_id, node in self.local_graph.items()
            if node.has_label(label)
        ]
        return {
            "nodes": nodes,
            "edges": [],
            "error": None,
            "stats": {"node_count": len(nodes), "edge_count": 0},
        }

    def _relationship_to_cytoscape(self, source_id, target_id, rel):
        """Converts a py2neo Relationship to a Cytoscape-compatible dict."""
        rel_type = type(rel).__name__
        edge_id = f"{source_id}_{target_id}_{rel_type}"

        props = {k: str(v) for k, v in rel.items()}

        return {
            "data": {
                "id": str(edge_id),
                "source": str(source_id),
                "target": str(target_id),
                "label": str(rel_type),
                **props,
            }
        }

    def _get_relationships_by_type(self, relationship_type):
        """Gets all relationships of a specific type."""
        nodes, edges, node_ids = [], [], set()

        for rel in self.relationships:
            if type(rel).__name__ == relationship_type:
                source_id = self._get_node_id(rel.start_node)
                target_id = self._get_node_id(rel.end_node)

                for node_id, node_obj in [
                    (source_id, rel.start_node),
                    (target_id, rel.end_node),
                ]:
                    if node_id not in node_ids:
                        node_ids.add(node_id)
                        nodes.append(self._node_to_cytoscape(node_id, node_obj))

                edges.append(self._relationship_to_cytoscape(source_id, target_id, rel))

        return {
            "nodes": nodes,
            "edges": edges,
            "error": None,
            "stats": {"node_count": len(nodes), "edge_count": len(edges)},
        }

    def _get_node_relationships(self, node_id_to_find):
        """Gets a specific node and all its relationships."""
        nodes, edges, node_ids = [], [], set()

        for rel in self.relationships:
            source_id = self._get_node_id(rel.start_node)
            target_id = self._get_node_id(rel.end_node)

            if source_id == node_id_to_find or target_id == node_id_to_find:
                for node_id, node_obj in [
                    (source_id, rel.start_node),
                    (target_id, rel.end_node),
                ]:
                    if node_id not in node_ids:
                        node_ids.add(node_id)
                        nodes.append(self._node_to_cytoscape(node_id, node_obj))

                edges.append(self._relationship_to_cytoscape(source_id, target_id, rel))

        return {
            "nodes": nodes,
            "edges": edges,
            "error": None,
            "stats": {"node_count": len(nodes), "edge_count": len(edges)},
        }

    def _get_all_graph_data(self):
        """Returns the entire graph formatted for Cytoscape."""
        nodes = [
            self._node_to_cytoscape(node_id, node)
            for node_id, node in self.local_graph.items()
        ]
        edges = [
            self._relationship_to_cytoscape(
                self._get_node_id(r.start_node), self._get_node_id(r.end_node), r
            )
            for r in self.relationships
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "error": None,
            "stats": {"node_count": len(nodes), "edge_count": len(edges)},
        }

    def _node_to_cytoscape(self, node_id, node):
        """Converts a py2neo Node to a Cytoscape-compatible dict."""
        node_type = list(node.labels)[0] if node.labels else "Unknown"
        props = {k: str(v) for k, v in node.items()}
        return {
            "data": {
                "id": str(node_id),
                "label": str(node.get("name", node_id)),
                "type": str(node_type),
                **props,
            },
            "classes": str(node_type).lower(),
        }

    def _get_node_id(self, node):
        """Generates a unique ID for a node for demo purposes."""
        for node_id, n in self.local_graph.items():
            if n == node:
                return node_id
        return str(node.identity)


# Initialize graph database
graph_db = GraphDatabase()


# Sample data for the landing page
def create_sample_data():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    data = {
        "date": dates,
        "value": np.cumsum(np.random.randn(100)) + 100,
        "category": np.random.choice(["A", "B", "C"], 100),
    }
    return pd.DataFrame(data)


# Create sample data
df = create_sample_data()


# Global constants
SAMPLE_QUERIES = [
    "MATCH (n) RETURN n",
    "MATCH (p:Person) RETURN p",
    "MATCH (p:Person)-[r:WORKS_WITH]->(p2:Person) RETURN p, r, p2",
    "MATCH (p:Person)-[r:REPORTS_TO]->(p2:Person) RETURN p, r, p2",
    "MATCH (p:Person)-[r:COLLABORATES_WITH]->(o:Organization) RETURN p, r, o",
    "MATCH (p:Person)-[r:LEADS]->(proj:Project) RETURN p, r, proj",
    "MATCH (c:Person {name: 'Charlie'})-[r]-(n) RETURN c, r, n",
    "CREATE (p:Person {name: 'NewPerson', age: 25, role: 'Analyst'})",
    "DELETE ALL",
]


# Helper Functions
def create_stat_cards():
    """Creates the statistics cards for the landing page."""
    stats_data = [
        {
            "label": "Total Data Points",
            "value": "1,234",
            "icon": "radix-icons:bar-chart",
            "color": "blue",
        },
        {
            "label": "Active Users",
            "value": "567",
            "icon": "radix-icons:person",
            "color": "green",
        },
        {
            "label": "Charts Created",
            "value": "89",
            "icon": "radix-icons:pie-chart",
            "color": "orange",
        },
        {
            "label": "Data Sources",
            "value": "12",
            "icon": "radix-icons:database",
            "color": "purple",
        },
    ]

    cards = []
    for stat in stats_data:
        card = dmc.Paper(
            p="md",
            withBorder=True,
            radius="md",
            children=[
                dmc.Group(
                    justify="space-between",
                    children=[
                        dmc.Stack(
                            gap=0,
                            children=[
                                dmc.Text(stat["label"], size="xs", c="gray"),
                                dmc.Text(stat["value"], size="xl", fw="bold"),
                            ],
                        ),
                        DashIconify(
                            icon=stat["icon"],
                            width=30,
                            color=f"var(--mantine-color-{stat['color']}-6)",
                        ),
                    ],
                ),
            ],
        )
        cards.append(card)
    return dmc.SimpleGrid(cols=4, children=cards)


def create_feature_cards():
    """Creates the feature cards for the landing page."""
    features_data = [
        {
            "title": "Fast & Responsive",
            "icon": "radix-icons:lightning-bolt",
            "color": "blue",
            "description": "Built with modern web technologies for lightning-fast performance and smooth interactions.",
        },
        {
            "title": "Beautiful UI",
            "icon": "radix-icons:palette",
            "color": "green",
            "description": "Modern and clean interface designed with Mantine components for the best user experience.",
        },
        {
            "title": "Highly Customizable",
            "icon": "radix-icons:gear",
            "color": "orange",
            "description": "Easily customize charts, themes, and layouts to match your specific needs and branding.",
        },
    ]

    cards = []
    for feature in features_data:
        card = dmc.Paper(
            p="xl",
            withBorder=True,
            radius="md",
            children=[
                dmc.Stack(
                    align="center",
                    gap="md",
                    children=[
                        DashIconify(
                            icon=feature["icon"],
                            width=40,
                            color=f"var(--mantine-color-{feature['color']}-6)",
                        ),
                        dmc.Title(feature["title"], order=3, size="h4"),
                        dmc.Text(feature["description"], ta="center", c="gray"),
                    ],
                )
            ],
        )
        cards.append(card)
    return dmc.SimpleGrid(cols=3, children=cards)


# App shell layout
def create_app_shell():
    return dmc.MantineProvider(
        theme={
            "colorScheme": "light",
            "primaryColor": "indigo",
            "components": {
                "Button": {"styles": {"root": {"fontWeight": 400}}},
                "Alert": {"styles": {"title": {"fontWeight": 500}}},
                "AvatarGroup": {"styles": {"truncated": {"fontWeight": 500}}},
            },
        },
        children=[
            dmc.AppShell(
                padding="md",
                id="app-shell",
                header={"height": 60},
                navbar={
                    "width": 250,
                    "breakpoint": "sm",
                    "collapsed": {"mobile": True, "desktop": False},
                },
                children=[
                    dmc.AppShellHeader(
                        children=[
                            dmc.Group(
                                justify="space-between",
                                align="center",
                                h="100%",
                                px="md",
                                children=[
                                    dmc.Group(
                                        children=[
                                            dmc.Burger(
                                                id="burger", size="sm", opened=False
                                            ),
                                            dmc.Title("Graph Dash", order=3),
                                        ]
                                    ),
                                    dmc.Avatar(
                                        src="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=256&q=80",
                                        radius="xl",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dmc.AppShellNavbar(
                        id="navbar",
                        p="md",
                        children=[
                            dmc.Stack(
                                children=[
                                    dmc.NavLink(
                                        label="Dashboard",
                                        leftSection=DashIconify(
                                            icon="radix-icons:dashboard", width=16
                                        ),
                                        id="nav-dashboard",
                                    ),
                                    dmc.NavLink(
                                        label="Graph Visualizer",
                                        leftSection=DashIconify(
                                            icon="radix-icons:share-2", width=16
                                        ),
                                        id="nav-graph",
                                    ),
                                ]
                            ),
                        ],
                    ),
                    dmc.AppShellMain(children=html.Div(id="page-content")),
                ],
            ),
        ],
    )


def create_landing_page():
    df = create_sample_data()
    line_fig = px.line(df, x="date", y="value", title="Time Series Data")
    line_fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))

    bar_fig = px.bar(
        df.groupby("category")["value"].mean().reset_index(),
        x="category",
        y="value",
        title="Average by Category",
    )
    bar_fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))

    return dmc.Container(
        size="lg",
        children=[
            dmc.Stack(
                gap="xl",
                children=[
                    dmc.Title(
                        "Welcome to Graph Dash Visualizer",
                        order=1,
                        size="h1",
                        ta="center",
                        mt=50,
                    ),
                    dmc.Text(
                        "A powerful dashboard for visualizing and analyzing your data with beautiful charts and interactive components.",
                        size="lg",
                        ta="center",
                        c="gray",
                        maw=600,
                        mx="auto",
                    ),
                    dmc.Group(
                        justify="center",
                        mt="xl",
                        children=[
                            dmc.Button(
                                "Get Started",
                                size="lg",
                                leftSection=DashIconify(
                                    icon="radix-icons:play", width=20
                                ),
                            ),
                            dmc.Button(
                                "View Documentation",
                                variant="outline",
                                size="lg",
                                leftSection=DashIconify(
                                    icon="radix-icons:book-open", width=20
                                ),
                            ),
                        ],
                    ),
                    create_stat_cards(),
                    dmc.SimpleGrid(
                        cols=2,
                        children=[
                            dmc.Paper(
                                withBorder=True,
                                radius="md",
                                p="md",
                                children=dcc.Graph(
                                    figure=line_fig, config={"displayModeBar": False}
                                ),
                            ),
                            dmc.Paper(
                                withBorder=True,
                                radius="md",
                                p="md",
                                children=dcc.Graph(
                                    figure=bar_fig, config={"displayModeBar": False}
                                ),
                            ),
                        ],
                    ),
                    dmc.Title("Features", order=2, ta="center", mt=50, mb=30),
                    create_feature_cards(),
                ],
            )
        ],
    )


def create_graph_visualizer():
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
                                                data=SAMPLE_QUERIES,
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
                                value="MATCH (n) RETURN n",
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
                                dmc.Paper(
                                    p="md",
                                    withBorder=True,
                                    children=dmc.Stack(
                                        gap="sm",
                                        children=[
                                            dmc.Text("Layout", fw="bold"),
                                            dmc.Select(
                                                data=[
                                                    {
                                                        "value": "cose",
                                                        "label": "Force-directed",
                                                    },
                                                    {
                                                        "value": "circle",
                                                        "label": "Circle",
                                                    },
                                                    {"value": "grid", "label": "Grid"},
                                                    {
                                                        "value": "random",
                                                        "label": "Random",
                                                    },
                                                    {
                                                        "value": "breadthfirst",
                                                        "label": "Hierarchical",
                                                    },
                                                ],
                                                value="cose",
                                                id="layout-select",
                                            ),
                                        ],
                                    ),
                                ),
                                dmc.Paper(
                                    p="md",
                                    withBorder=True,
                                    children=dmc.Stack(
                                        gap="sm",
                                        children=[
                                            dmc.Text("Query Results", fw="bold"),
                                            html.Div(id="query-stats"),
                                        ],
                                    ),
                                ),
                                dmc.Paper(
                                    p="md",
                                    withBorder=True,
                                    children=dmc.Stack(
                                        gap="sm",
                                        children=[
                                            dmc.Text("Selected Element", fw="bold"),
                                            html.Div(id="selected-info"),
                                        ],
                                    ),
                                ),
                            ],
                        ),
                    ),
                ],
            ),
            html.Div(id="error-display"),
        ],
    )


def get_cytoscape_stylesheet():
    return [
        {
            "selector": "node",
            "style": {
                "content": "data(label)",
                "width": "60px",
                "height": "60px",
                "font-size": "12px",
                "text-valign": "center",
                "text-halign": "center",
                "background-color": "#3498db",
                "color": "white",
                "text-outline-width": 2,
                "text-outline-color": "#3498db",
            },
        },
        {"selector": ".person", "style": {"background-color": "#e74c3c"}},
        {"selector": ".organization", "style": {"background-color": "#2ecc71"}},
        {"selector": ".project", "style": {"background-color": "#f39c12"}},
        {"selector": ".technology", "style": {"background-color": "#9b59b6"}},
        {
            "selector": "edge",
            "style": {
                "content": "data(label)",
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "line-color": "#95a5a6",
                "target-arrow-color": "#95a5a6",
                "font-size": "10px",
                "text-rotation": "autorotate",
            },
        },
        {
            "selector": ":selected",
            "style": {
                "background-color": "#34495e",
                "line-color": "#34495e",
                "target-arrow-color": "#34495e",
                "source-arrow-color": "#34495e",
            },
        },
    ]


# App layout
app.layout = create_app_shell()


# Callbacks
@callback(
    Output("page-content", "children"),
    [Input("nav-dashboard", "n_clicks"), Input("nav-graph", "n_clicks")],
)
def update_page_content(dashboard_clicks, graph_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return create_landing_page()

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "nav-graph":
        return create_graph_visualizer()
    return create_landing_page()


@callback(
    Output("app-shell", "navbar"),
    Input("burger", "opened"),
    State("app-shell", "navbar"),
)
def toggle_navbar(opened, navbar):
    navbar["collapsed"] = {"mobile": not opened, "desktop": False}
    return navbar


def _build_cypher_response(result):
    """Helper to build the elements, stats, and error components from a query result."""
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
    Input("execute-btn", "n_clicks"),
    Input("clear-graph-btn", "n_clicks"),
    State("cypher-input", "value"),
)
def execute_cypher_query(execute_clicks, clear_clicks, query):
    ctx = dash.callback_context
    button_id = (
        ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "initial_load"
    )

    if button_id == "clear-graph-btn":
        graph_db.initialize_sample_data()
        result = graph_db.execute_cypher("MATCH (n) RETURN n")
    elif button_id == "execute-btn" and query:
        result = graph_db.execute_cypher(query)
    elif button_id == "initial_load":
        result = graph_db.execute_cypher("MATCH (n) RETURN n")
    else:
        return dash.no_update

    return _build_cypher_response(result)


@callback(
    Output("cypher-input", "value"),
    Input("sample-queries", "value"),
    Input("clear-btn", "n_clicks"),
)
def update_query_input(sample_query, clear_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "clear-btn":
        return ""
    if button_id == "sample-queries" and sample_query:
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
    title = f"Type: {element_type}"
    if node_data:
        title += f" ({data.get('type', 'Unknown')})"

    core_properties = (
        {"id", "label", "type"} if node_data else {"id", "source", "target", "label"}
    )

    details = [dmc.Text(title, size="sm", fw="bold")]
    if node_data:
        details.extend(
            [
                dmc.Text(f"ID: {data.get('id', 'N/A')}", size="sm"),
                dmc.Text(f"Label: {data.get('label', 'N/A')}", size="sm"),
            ]
        )
    else:
        details.extend(
            [
                dmc.Text(f"Source: {data.get('source', 'N/A')}", size="sm"),
                dmc.Text(f"Target: {data.get('target', 'N/A')}", size="sm"),
                dmc.Text(f"Label: {data.get('label', 'N/A')}", size="sm"),
            ]
        )

    details.append(html.Hr())
    details.append(dmc.Text("Properties:", size="sm", fw="bold"))

    properties = {k: v for k, v in data.items() if k not in core_properties}
    if properties:
        details.extend(
            [dmc.Text(f"{k}: {v}", size="xs") for k, v in properties.items()]
        )
    else:
        details.append(dmc.Text("No additional properties.", size="xs", c="gray"))

    return details


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
