from src.agents.report_generation_agent import export_to_pdf

sample_report = """# Research Report: transformers

## Executive Summary
This is a test paragraph with a reasonably long line to check wrapping behavior works correctly.

## References
- https://fake2.com (Self-attention definition)
"""

export_to_pdf(sample_report, "test_output.pdf")
print("PDF created successfully")