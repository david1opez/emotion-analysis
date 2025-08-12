FROM public.ecr.aws/lambda/python:3.12

# Copy the requirements file
COPY requirements.txt  ${LAMBDA_TASK_ROOT}

# Install the dependencies
RUN pip install -r requirements.txt

# Copy pre-downloaded model/tokenizer into container
COPY model/tokenizer /opt/model/tokenizer

# Copy all the files in ./src
COPY src/* ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD ["lambda_function.lambda_handler"]
