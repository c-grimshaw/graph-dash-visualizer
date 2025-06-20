# Graph Dash Visualizer

A modern, responsive dashboard application built with Dash and Mantine components for data visualization and analysis.

## Features

- **Modern UI**: Built with Dash Mantine Components for a beautiful, responsive interface
- **App Shell**: Professional layout with sidebar navigation and header
- **Interactive Charts**: Sample visualizations using Plotly
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Navigation**: Multi-page application with smooth transitions

## Getting Started

### Prerequisites

- Python 3.13+
- uv (Python package manager)

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   uv sync
   ```

### Running the Application

```bash
uv run python app.py
```

The application will be available at `http://localhost:8050`

## Project Structure

- `app.py` - Main application file with Dash app and Mantine components
- `pyproject.toml` - Project configuration and dependencies
- `README.md` - This file

## Dependencies

- `dash` - Web application framework
- `dash-mantine-components` - Mantine UI components for Dash
- `dash-iconify` - Icon components
- `plotly` - Interactive charts and graphs
- `pandas` - Data manipulation
- `numpy` - Numerical computing

## Pages

- **Dashboard**: Landing page with sample charts and statistics
- **Analytics**: Placeholder for analytics features
- **Settings**: Placeholder for application settings
- **Documentation**: Placeholder for documentation

## Customization

The app uses Mantine's theming system. You can customize colors, fonts, and other design elements by modifying the theme configuration in the `create_app_shell()` function.
