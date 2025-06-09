import joblib
from explainerdashboard import ExplainerDashboard
from explainerdashboard.explainers import ClassifierExplainer

# Load explainer
explainer = ClassifierExplainer.from_file("model/explainer.pkl")

# Launch dashboard
db = ExplainerDashboard(explainer,
                        title="Loan Approval Prediction Dashboard",
                        whatif=True,
                        shap_interaction=True,
                        decision_trees=True,
                        importances=True)
db.run()

