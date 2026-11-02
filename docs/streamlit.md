# Building Apps with Streamlit

Streamlit is a Python library for turning scripts into interactive web apps without writing HTML, CSS, or JavaScript. You run an app with the command "streamlit run" followed by the file name, and it opens in the browser at localhost:8501.

Streamlit reruns the entire script from top to bottom every time a user interacts with the page, such as clicking a button or typing in a box. This makes apps simple to write, but it also means regular variables are reset on every rerun.

Session state is how Streamlit remembers information between reruns. Values stored in st.session_state, such as a chat history or a quiz score, survive when the script runs again.

Caching prevents slow work from repeating on every rerun. The decorator st.cache_data is used for data like loaded files or API results, while st.cache_resource is used for shared objects like database connections and machine learning models.

Streamlit has built-in chat elements for AI apps. The st.chat_input widget creates a message box, st.chat_message creates chat bubbles for the user and assistant, and st.write_stream displays a streaming response as it arrives.