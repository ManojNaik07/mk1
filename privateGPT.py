import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout, QRadioButton
from PyQt5.QtGui import QPalette, QColor, QTextCursor
from PyQt5.QtCore import Qt
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import chromadb
import os
import argparse

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

class PrivateGPTApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('PRIVATEGPT GUI')

        
        self.dark_palette = QPalette()
        self.dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        self.dark_palette.setColor(QPalette.WindowText, Qt.white)
        self.dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        self.dark_palette.setColor(QPalette.ButtonText, Qt.white)
        self.setPalette(self.dark_palette)

    
        layout = QVBoxLayout(self)

        
        private_label = QLabel('PRIVATEGPT')
        private_label.setStyleSheet("color: #00BFFF; font-size: 36px; font-weight: bold; text-align: center")
        layout.addWidget(private_label, alignment=Qt.AlignCenter)

    
        self.query_label = QLabel('Enter a query:')
        self.query_label.setStyleSheet("color: #00BFFF")
        layout.addWidget(self.query_label, alignment=Qt.AlignTop)


        self.query_entry = QLineEdit()
        self.query_entry.setStyleSheet("border: 1px solid #ccc; padding: 10px; font-size: 14px; background-color: #333; color: white")
        layout.addWidget(self.query_entry)

    
        self.ask_button = QPushButton('Ask')
        self.ask_button.setStyleSheet("background-color: #008000; color: white; border: 1px solid #ccc; padding: 10px; font-size: 16px")
        self.ask_button.clicked.connect(self.ask_question)


        h_layout = QHBoxLayout()
        h_layout.addStretch(1)
        h_layout.addWidget(self.ask_button)
        h_layout.addStretch(1)
        layout.addLayout(h_layout)


        self.result_text = QTextEdit()
        self.result_text.setStyleSheet("border: 1px solid #ccc; padding: 10px; background-color: #333; color: white")
        layout.addWidget(self.result_text)

        
        self.dark_radio = QRadioButton("Dark Theme")
        self.light_radio = QRadioButton("Light Theme")

    
        self.dark_radio.setChecked(True)
        self.light_radio.setChecked(False)

    
        self.dark_radio.toggled.connect(self.set_dark_theme)
        self.light_radio.toggled.connect(self.set_light_theme)

    
        layout.addWidget(self.dark_radio)
        layout.addWidget(self.light_radio)

        self.setLayout(layout)

        self.args = self.parse_arguments()

        self.initialize_backend()

    def ask_question(self):
        query = self.query_entry.text().strip()
        if query:
            res = self.qa(query)
            answer, docs = res['result'], [] if self.args.hide_source else res['source_documents']
            result_str = f"Question:\n{query}\n\nAnswer:\n{answer}\n\n"
            for document in docs:
                result_str += f"\nSource: {document.metadata['source']}:\n{document.page_content}\n"
            self.result_text.setPlainText(result_str)

    def initialize_backend(self):
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=persist_directory)
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings,
                    client_settings=CHROMA_SETTINGS, client=chroma_client)
        retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

        callbacks = []

        if model_type == "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch,
                           callbacks=callbacks, verbose=False, n_threads=16)
        elif model_type == "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj',
                          n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        else:
            raise Exception(f"Model type {model_type} is not supported. "
                            f"Please choose one of the following: LlamaCpp, GPT4All")

        self.qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                              return_source_documents=True)

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                     'using the power of LLMs.')
        parser.add_argument("--hide-source", "-S", action='store_true',
                            help='Use this flag to disable printing of source documents used for answers.')

        parser.add_argument("--mute-stream", "-M",
                            action='store_true',
                            help='Use this flag to disable the streaming StdOut callback for LLMs.')

        return parser.parse_args()

    def set_dark_theme(self):
        if self.dark_radio.isChecked():
            self.setPalette(self.dark_palette)
            self.query_entry.setStyleSheet("border: 1px solid #ccc; padding: 10px; font-size: 14px; background-color: #333; color: white")
            self.result_text.setStyleSheet("border: 1px solid #ccc; padding: 10px; background-color: #333; color: white")

    def set_light_theme(self):
        if self.light_radio.isChecked():
            self.setPalette(QPalette()) 
            self.query_entry.setStyleSheet("border: 1px solid #ccc; padding: 10px; font-size: 14px; background-color: white; color: black")
            self.result_text.setStyleSheet("border: 1px solid #ccc; padding: 10px; background-color: white; color: black")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    private_gpt_app = PrivateGPTApp()
    private_gpt_app.show()
    sys.exit(app.exec_())




















