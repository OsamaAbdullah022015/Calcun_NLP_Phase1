# Use the Python 3.11.4 base image
FROM python:3.11.4

# Install git and other required dependencies
RUN apt-get update && apt-get install -y git
RUN apt-get install default-jre -y

# Install system dependencies for detectron2
RUN apt-get install -y --no-install-recommends tesseract-ocr libtesseract-dev poppler-utils

# Making working directory
RUN mkdir /app

#Making output directory
RUN mkdir /app/output

# Set a working directory inside the container
WORKDIR /app

# Copy the model script and requirements into the container
COPY requirements.txt /app/
COPY gen_kg.ipynb /app/
COPY ./utils /app/utils
COPY ./nltk_data /root/nltk_data
COPY ./metadata /app/metadata

#Installing other libraries
RUN pip install -r requirements.txt
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# RUN pip install pytesseract
# RUN pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.4#egg=detectron2'

# Download the en_core_web_sm-3.6.0 .whl file
RUN wget https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0-py3-none-any.whl
RUN wget https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.7.3/en_core_web_trf-3.7.3-py3-none-any.whl
RUN wget https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl

# Install the downloaded .whl file using pip
RUN pip install -U en_core_web_sm-3.6.0-py3-none-any.whl
RUN pip install -U en_core_web_trf-3.7.3-py3-none-any.whl
RUN pip install -U en_core_web_lg-3.7.1-py3-none-any.whl

# Expose the Jupyter Notebook port
EXPOSE 8888 9000

# Start Jupyter Notebook when the container runs
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
