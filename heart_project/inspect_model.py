# inspect_model.py
import joblib
model = joblib.load("heart_model.pkl")
print("MODEL TYPE:", type(model))
# sklearn models usually have attribute feature_names_in_
fn = getattr(model, "feature_names_in_", None)
print("feature_names_in_:", fn)
# If not present, print a short repr
try:
    print("n_features_in_:", getattr(model, "n_features_in_", None))
except Exception as e:
    print("n_features_in_ error:", e)
