from kogpt2_train import Transformers_kogpt
from kogpt2_run import Chatbot
from kobert_train import Transformers_kobert
from bertClassifier import BERTClassifier




if __name__ == '__main__':
    # gpt = Transformers_kogpt()
    # gpt.hook()
    chat = Chatbot()
    chat.execute_model()
    # bert = Transformers_kobert()
    # bert.hook()
    
   