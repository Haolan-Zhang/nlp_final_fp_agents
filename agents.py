from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chat_models import ChatOpenAI


class BaseAgent:
    def __init__(self, retriever, template):
        self.retriever = retriever
        self.prompt = ChatPromptTemplate.from_template(template)
        self.llm = ChatOpenAI()
        self.rag_chain = (
            {"context": self.retriever, "input": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def invoke(self, query):
        return self.rag_chain.invoke(query)


class CodeSearcherAgent(BaseAgent):
    def __init__(self, retriever):
        searcher_template = """
        Based on the {user’s design focus}
        and the {current floor plan}, identify the
        most relevant sections of the building
        code that apply. Return the code sections.

        Input:
        {input}

        Retrieved Context:
        {context}
        """
        super().__init__(retriever, searcher_template)


class CodeExaminerAgent(BaseAgent):
    def __init__(self, retriever):
        examiner_template = """
        Based on the {provided building code
        sections} and the {current floor plan},
        evaluate whether the design complies
        with the applicable requirements. If it
        does not, identify the issues and provide
        code-based suggestions and brief explanations.

        Input:
        {input}

        Relevant Building Code Context:
        {context}

        Assessment:
        """
        super().__init__(retriever, examiner_template)


class LeadDesignerAgent(BaseAgent):
    def __init__(self, retriever):
        designer_template = """
        Based on the {code-based suggestions
        and explanations}, and the {current
        floor plan}, generate a detailed adjustment
        plan to bring the floor plan into compliance with building code requirements.

        For each room that requires modification, provide:
        - room index
        - adjustment type
        - required fields (e.g., percent change or room type)

        Return the output as a JSON array.
        Do not include fields that are not required for the specified adjustment type.

        Input:
        {input}

        Context:
        {context}

        Adjustment Plan:
        """
        super().__init__(retriever, designer_template)
