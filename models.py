# one jp model
# one multilingual model


# use classes: EXAMPLE below
# # models/base_model.py
class BaseModel:
    def __init__(self, model_name):
        self.model_name = model_name
        # Here you could initialize API client if needed
    
    def prompt(self, text):
        """
        Send a prompt to the model and return its response.
        """
        raise NotImplementedError("This should be implemented in subclasses")
    
# models/model_a.py
# from .base_model import BaseModel

class ModelA(BaseModel):
    def prompt(self, text):
        # Imagine this sends 'text' to Model A and returns the response
        response = f"ModelA says: {text[:10]}..."  # placeholder
        return response
    
# main.py
# from models.model_a import ModelA
# from models.model_b import ModelB

def main():
    prompt_text = "Explain quantum physics in one sentence."

    # Initialize models
    models = [ModelA("A"), ModelB("B")]

    # Run prompt for each
    for model in models:
        output = model.prompt(prompt_text)
        print(f"{model.model_name} output: {output}")

if __name__ == "__main__":
    main()