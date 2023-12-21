import os
import shutil
import gradio as gr

def UploadFile():
    # Function to handle PDF file upload and save
    def handle_upload(pdf_file, model):
        directory = 'uploads'
        if not os.path.exists(directory):
            os.makedirs(directory)

        if pdf_file and model:
            target_path = "./uploads/" + pdf_file.name.split('/')[-1]
            shutil.copyfile(pdf_file, target_path)
            print(f"PDF file '{pdf_file.name}' has been uploaded and saved to {target_path}")
            # Ensure the directory exists before writing the file
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Write the selected model to a file
            model_path = os.path.join(directory, 'model.txt')
            # Update the global variable with the selected model
            with open(model_path, 'w') as file:
                # Write content to the file
                file.write(model)
            print(f"Selected model: {model}")

            # Write the selected file name to a file
            file_name_path = os.path.join(directory, 'file_name.txt')
            # Update the global variable with the selected model
            with open(file_name_path, 'w') as file:
                # Write content to the file
                file.write(target_path)
            return "File uploaded and model selected"

        return "No file uploaded and no model selected."

    # Create a Gradio interface for PDF file upload
    iface = gr.Interface(
        fn=handle_upload,
        inputs=[
            gr.File(file_types=['.pdf']),
            gr.Dropdown(
                ["en_core_web_sm", "en_core_web_trf", "en_core_web_lg"],
                value="en_core_web_sm",
                label="Choose the model to be used."
            )
        ],
        outputs="text",
        live=False
    )

    # Launch the Gradio interface
    iface.launch(server_name='0.0.0.0', server_port=9000)

