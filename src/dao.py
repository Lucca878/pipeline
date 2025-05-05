class history:
    
    def __init__(self, statement, parapharsed):
        self.statement = statement
        self.parapharsed = parapharsed

        self.history_parapharsed = []

    def update_statement(self, statement):
        self.statement = statement

    def add_parapharsed(self, parapharsed):
        self.history_parapharsed.append(parapharsed)

class statement:
    def __init__(self, statement, classfication, score):
        self.statement = statement
        self.classfication = classfication
        self.score = score

class parapharsed_obj:
    def __init__(self, paraphrased_text, classfication, score):
        self.paraphrased_text = paraphrased_text
        self.classfication = classfication
        self.score = score