from src.predict_load_bert import predict_report

report = """
The left kidney was removed.
The right kidney is present.
The liver appears present.
The spleen is present.
"""

result = predict_report(report)

print(result)