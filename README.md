# Chat with PDF using LLMs and RAG

This code is written based on this [tutorial](https://takehomes.com/library/developers/intro-to-ai).  
The Readme and the boilerplate code is written by the tutorial author, a development environment was provided, I filled in the empty code stubs with functional code and learned the concepts.

## Overview

In this lesson, we will build a chat bot that can answer questions about a provided PDF file. The goal of this lesson is to expose you to many concepts in AI for working with Large Language Models (LLMs). You are not expected to know how to implement all of the coding tasks completely, but you should try to go as far as you can and then compare your solution to the provided solution, or just copy and paste the provided solution to move on.

To get started, you will first need to create an [OpenAI API key](https://platform.openai.com/account/api-keys). Next, open the `.env` file and paste your key into the file.

```
OPENAI_API_KEY="<your-key-here>"
```

> Note: OpenAI has switched to a [pre-paid billing model](https://help.openai.com/en/articles/8264644-what-is-prepaid-billing), so you will need to have credits in your account to use your API key. After adding credits to your account, it may take up to 10 minutes for the credits to be applied.

## Lesson Plan

<details>
<summary>Section 1: Understanding the Limitations of LLMs</summary>

### Section 1: Understanding the Limitations of LLMs

While LLMs are very powerful tools and getting more powerful each day, they still have some fundamental limitations. The first limitation is that they are typically trained on data from years ago.

> `gpt-3.5-turbo` is trained on data only up until [September 2021](https://platform.openai.com/docs/models/gpt-3-5-turbo)

- What if we need to ask a question about more recent data?
- Or about something that the LLM was not trained on?

Additionally, the text input to LLMs are transformed into [tokens](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them) and every LLM model has a token limit.

> `gpt-3.5-turbo` has a limit of about [16k tokens](https://platform.openai.com/docs/models/gpt-3-5-turbo)

#### Task 0 — Hitting the Token Limit

You can see this limitation for yourself. Take a look at the `limitations.py` file. In that file, you can see that we try to load up a large text file, and ask ChatGPT to summarize it.

1. Make sure your OpenAI API key has been added to `.env`
2. Run the `1-install` task and wait for the dependencies to install
3. After the task above completes, run the shell with `2-dev`

Once you have a Poetry shell, run

```bash
python limitations.py
```

- ❓What is the error that you get?

#### Now that you know about the limitations of LLMs, can you think of ways to overcome them?

<details>
<summary>💡 Reveal answers</summary>

- We can solve the September 2021 data cutoff problem by feeding additional data to ChatGPT
- We can overcome the token limit problem by breaking our request into multiple smaller requests
</details>

</details>

---

<details>
<summary>Section 2: Using LangChain to Overcome the Limitations</summary>

### Section 2: Using LangChain to Overcome the Limitations

Now that we understand the limitations of working with LLMs and the techniques for overcoming them, let's go over how we will actually do that.

One approach is to write the code to solve this problem ourselves. We can build a system that takes a large text file, split it into chunks, store it in a database so we can query it for certain keywords, and then use it to pass additional context to ChatGPT.

However, it turns out that this is a pretty common pattern for working with AI and LLMs, so companies like [LangChain](https://www.langchain.com) have already done the hard work for us. You can think of LangChain as an AI developer toolkit that has useful `components` that you can string together into a `chain`. Once you have created a `chain`, you can call `invoke()` on the `chain` to execute it. Read more about LangChain [here](https://python.langchain.com/docs/get_started/introduction) on your own time.

We will need one additional component to store the text data after we process it—something called a `vector database`. In this lesson, we will be using an open-source vector database called [Chroma](https://www.trychroma.com), but there are many others like [Milvus](https://milvus.io) and [Pinecone](https://www.pinecone.io). If you don't know what a vector database is, you can read more about it [here](https://codelabs.milvus.io) on your own time.

<details>
<summary>TL;DR what's a vector database?</summary>

A vector database allows us to store unstructured data into a high-dimensional vector space. We use an `embedding function` to convert that data into a long array of integers (our _n_-dimensional vector). Once in vector space, we can use `semantic search` to query _not just_ with keywords but with `semantic meaning`. That is, a vector database can return results containing the word `python` _even if_ we only search for the word `snake`. Moreover, it can also distinguish between `python` used in a sentence to mean an animal vs. `python` used as a programming language.

</details>

#### You might ask why don't we just upload our text/PDF files directly to ChatGPT?

Well, ChatGPT does not support PDF files out of the box. So we need to transform the PDF into text. But if we have a sufficiently large PDF, then the amount of text will not fit inside the token limit. Furthermore, it's very inefficient to embed an entire PDF's worth of text into every ChatGPT message.

This method of only _retrieving_ the relevant content related to our question is called **Retrieval Augmented Generation** (**RAG**). You can watch a video about RAG [here](https://research.ibm.com/blog/retrieval-augmented-generation-RAG) on your own time.

</details>

---

<details>
<summary>Section 3: Data Ingest Pipeline</summary>

### Section 3: Data Ingest Pipeline

In this section, we'll be going over a series of steps to implement the system described above. You will be given the opportunity to code up the solution yourself, but we will also provide the answer in case you need some help.

### Step 1. Processing the PDF file

Open the `ingest.py` file and take a look at the code for the main block. We see that in Step 1, the program is trying to load the PDF file located in `data/` and parse out the pages and metadata.

In `parse_pdf()`, we extract the metadata and then the pages.

#### Task I — Implement `extract_metadata_from_pdf`

Your first task is to implement this function. You can use the [PyPDF](https://pypdf2.readthedocs.io/en/1.27.12/) library to get the metadata from the file. You may find the `PdfFileReader` class and its `getDocumentInfo()` method useful.

<details>
<summary>💡 Reveal solution</summary>

```python
def extract_metadata_from_pdf(file_path: str) -> dict:
    """
    Extracts the PDF file metadata.

    :param file_path: The path to the PDF file.
    :return: A dictionary containing the title and creation_date.
    """
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF4.PdfFileReader(pdf_file)
        metadata = reader.getDocumentInfo()
        return {
            "title": metadata.get("/Title", "").strip(),
            "creation_date": metadata.get("/CreationDate", "").strip(),
        }
```

</details>

#### Task II — Implement `extract_pages_from_pdf`

Next, you'll need to extract the text from each page of the PDF. You can use [pdfplumber](https://pypi.org/project/pdfplumber/) to do this. Use Python's `enumerate()` to loop through the pages of the PDF. Return a list of the tuple `(page_number, extracted_text)` as indicated by the return type.

<details>
<summary>💡 Reveal solution</summary>

```python
def extract_pages_from_pdf(file_path: str) -> List[Tuple[int, str]]:
    """
    Extracts the text from each page of the PDF.

    :param file_path: The path to the PDF file.
    :return: A list of tuples containing the page number and the extracted text.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with pdfplumber.open(file_path) as pdf:
        pages = []
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text.strip():  # Check if extracted text is not empty
                pages.append((page_num + 1, text))
    return pages
```

</details>

### Step 2. Splitting the pages into chunks

It may be the case that a full page of text is too large to fit inside a context window for a particular LLM. So we'll need to split our pages down even futher into smaller chunks. Luckily, LangChain provides a utility to help us with that task.

#### Task III — Implement `text_to_docs`

Fill in the code for this function, which takes our raw PDF pages and turns it into a list of `Documents`.

You will first need to split the text into chunks using LangChain's [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter). Use `1000` for the `chunk_size` and `200` for the `chunk_overlap`. If you want to read more about these parameters, check out this [thread](https://github.com/langchain-ai/langchain/issues/2026).

<details>
<summary>🖹 What is a Document?</summary>

A [document](https://js.langchain.com/docs/modules/data_connection/document_loaders/creating_documents) consists of a piece of text and optional metadata. The piece of text is what we interact with the language model, while the optional metadata is useful for keeping track of metadata about the document (such as the source).

</details>

Once you have these chunks, create a new [`Document`](https://api.python.langchain.com/en/latest/documents/langchain_core.documents.base.Document.html) containing the relevant `page_content` and `metadata`.

<details>
<summary>💡 Reveal solution</summary>

```python
def text_to_docs(
    text: List[Tuple[int, str]], metadata: Dict[str, str]
) -> List[Document]:
    """
    Converts a list of tuples of page numbers and extracted text to a list of Documents.
    """
    doc_chunks = []

    for page_num, page in text:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(page)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page_number": page_num,
                    "chunk": i,
                    "source": f"p{page_num}-{i}",
                    **metadata,
                },
            )
            doc_chunks.append(doc)

    return doc_chunks
```

</details>

### Step 3. Generate embeddings and store them in the DB

This is the final step in processing our PDF file. We now store our list of `Documents` into a vector database and save it to disk.

#### Task IV — Implement `store_chunks`

Using the [Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma) database wrapper in LangChain, create a vector database from our document chunks.

In order for us to insert the text data into the vector database, we need to first convert the text into a vector in semantic space. We do that using an [embedding function](https://platform.openai.com/docs/guides/embeddings). The wrapper for the OpenAI embedding function in LangChain is `OpenAIEmbeddings()`.

<details>
<summary>💡 Reveal solution</summary>

```python
def store_chunks(document_chunks: List[Document], collection_name: str, directory: str):
    """
    Creates a Chroma vector store and persists the documents to disk.

    :param document_chunks: A list of Documents to be stored.
    :param collection_name: Name of this collection.
    :param directory: Location on disk to persist the data.
    """
    embeddings = OpenAIEmbeddings()

    vector_store = Chroma.from_documents(
        document_chunks,
        embeddings,
        collection_name=collection_name,
        persist_directory=directory,
    )

    vector_store.persist()
```

</details>

#### Task V — Run the data ingest program

You can now run the program by using the `2-dev` task to open a Poetry shell. Make sure you have already ran the `1-install` task, and added your OpenAI API key to the `.env` file.

Once a Poetry shell has been opened, run the data ingest program:

```bash
python ingest.py
```

This program should take a few seconds to run. Have a look at the `data/chroma/` directory to see that your text chunks and its embedding have been persisted to disk.

</details>

---

<details>
<summary>Section 4: Using RAG with Chat</summary>

### Section 4: Using RAG with Chat

Now we can finally have some fun with all this work. Let's do a quick overview on where we are. We have ingested a PDF file, split it into pages, broken up those pages into manageable-sized chunks, and stored the chunks into our vector database (Chroma).

In the `chat.py` file, you will see that we setup the vector store before calling `make_chain()`. Note that in order for semantic search to work properly, we need to use the exact same embedding function we used for data ingestion in the constructor for `Chroma`. In our case, we are using `OpenAIEmbeddings()` as before.

After creating the chain and allocating a variable to store our chat history, we run a `while True` loop. In this loop, we wait for the user's question and then call `invoke()` on our chain, passing along the question and the chat history. Calling `invoke()` will finally make an API call to OpenAI to query ChatGPT. Finally, we parse the response, add it to the chat history, and display the results.

#### Task VI — Implement `make_chain`

Use the [`ChatOpenAI`](https://python.langchain.com/docs/integrations/chat/openai) constructor to get an instance of the specified LLM model. Next, create a [`ConversationalRetrievalChain`](https://api.python.langchain.com/en/latest/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html) from this model, using our `vector_store` as the retriever.

<details>
<summary>💡 Reveal solution</summary>

```python
def make_chain(model_name: str, vector_store: Chroma):
    """
    Creates a Chroma vector store and persists the documents to disk.

    :return: A ConversationalRetrievalChain
    """
    model = ChatOpenAI(
        model_name=model_name,
        temperature="0",
        # verbose=True
    )

    return ConversationalRetrievalChain.from_llm(
        model,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        # verbose=True,
    )
```

</details>

#### Task VII — Run the chat program and ask questions

Back in your Poetry shell, run the chat program:

```
python chat.py
```

This program should prompt you for a question. You can ask it any question about the PDF file. For example:

- What does the report say about the economic outlook of 2024?

Try a few more questions on your own.

#### (Optional) Task VIII — Turn on verbose mode

Add `verbose=True` to both `ChatOpenAI` and `ConversationalRetrievalChain` in the chain and re-run the program. Now when you ask questions, you will see what LangChain does in the background in order to implement RAG.

</details>

</details>

---

## Conclusion

In this lesson, you have learned how to work with LLMs, including where they tend to struggle. You have also learned how to use an AI toolkit like LangChain to help overcome some of these weaknesses. In doing so, you implement a basic RAG system that can answer questions about a PDF file containing recent economic data.

We hope that you have enjoyed this lesson!

## Resources

- [Chroma](https://www.trychroma.com)
- [LangChain — Getting Started](https://python.langchain.com/docs/get_started/introduction)
- [Vector Database 101](https://codelabs.milvus.io)
- [What is Retrieval Augmented Generation?](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)

## Special Thanks

The code for lesson was modified from [edrickdch/chat-pdf](https://github.com/edrickdch/chat-pdf).
