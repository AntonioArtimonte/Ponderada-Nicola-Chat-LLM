import streamlit as st
from openai import OpenAI
import nltk
from nltk.corpus import stopwords
import re
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

class IndustrialSafetyChatbot:
    def __init__(self):
        # Carregar variáveis de ambiente
        load_dotenv()

        # Configurar cliente OpenAI
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        # Documento base
        self.document = """
        Workshop rules and safety considerations:
        1. Only trained personnel are allowed to operate a lathe.
        2. Protective gear includes safety goggles, gloves, and ear protection.
        3. No loose clothing or jewelry should be worn near machinery.
        4. Always check that all safety guards are in place before starting any machine.
        5. Emergency stops should always be identified before use.
        """

        # Certifique-se de que as stopwords do NLTK estão baixadas uma vez
        nltk.download("stopwords", quiet=True)

        # Carregar vetor de documentos
        self.vector_store = self._load_document()

    def _load_document(self):
        # Dividir o documento em pedaços para recuperação eficiente
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(self.document)

        # Criar embeddings e índice FAISS
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(chunks, embeddings)
        return vector_store

    def extract_keywords(self, query):
        """Extrai palavras-chave de uma consulta para reduzir o uso de tokens."""
        stop_words = set(stopwords.words("english"))
        words = re.findall(r"\w+", query.lower())
        keywords = [word for word in words if word not in stop_words]
        return " ".join(keywords)

    def generate_response(self, query):
        """Gera uma resposta utilizando RAG."""
        # Recuperar trechos relevantes do documento
        results = self.vector_store.similarity_search(query, k=2)
        context = "\n".join([result.page_content for result in results])

        # Construir o prompt
        prompt = f"Based on the following context:\n\n{context}\n\nAnswer the question: {query}"

        # Gerar resposta com OpenAI
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in industrial safety."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

# Função principal da interface Streamlit
def main():
    st.title("Chatbot de Normas de Segurança Industrial")

    # Inicializar chatbot
    chatbot = IndustrialSafetyChatbot()

    # Inicializar histórico de mensagens
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibir histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Aceitar entrada do usuário
    if prompt := st.chat_input("Qual é a sua dúvida sobre normas de segurança?"):
        # Extrair palavras-chave (opcional)
        keywords = chatbot.extract_keywords(prompt)
        st.write(f"Palavras-chave extraídas: {keywords}")

        # Adicionar mensagem do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gerar e exibir resposta do assistente
        with st.chat_message("assistant"):
            response = chatbot.generate_response(prompt)
            st.markdown(response)

        # Adicionar resposta ao histórico
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
