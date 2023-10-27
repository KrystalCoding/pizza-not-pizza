import streamlit as st
from app_pages.multipage import MultiPage

from app_pages.page_summary import page_summary_body
from app_pages.page_visualizer import page_visualizer_body
from app_pages.page_pizza_detector import page_pizza_detector_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_ml_performance import page_ml_performance_metrics

app = MultiPage(app_name="Pizza Detector")

app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Visualiser", page_visualizer_body)
app.add_page("Pizza Detection", page_pizza_detector_body)
app.add_page("Project Hypothesis", page_project_hypothesis_body)
app.add_page("ML Performance Metrics", page_ml_performance_metrics)

app.run()
