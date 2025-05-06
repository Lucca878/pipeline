class history:
    
    def __init__(self, statement, paraphrased):
        self.statement = statement
        self.paraphrased = paraphrased

        self.history_paraphrased = []

    def update_statement(self, statement):
        self.statement = statement

    def add_paraphrased(self, paraphrased):
        self.history_paraphrased.append(paraphrased)

class statement:
    def __init__(self, statement, classification, score):
        self.statement = statement
        self.classification = classification
        self.score = score

class paraphrased_obj:
    def __init__(self, paraphrased_text, classification, score):
        self.paraphrased_text = paraphrased_text
        self.classification = classification
        self.score = score