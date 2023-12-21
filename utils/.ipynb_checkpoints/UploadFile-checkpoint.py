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
            target_path = "./uploads" + '/test.pdf'
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
            return "File uploaded and model selected"

        return "No file uploaded and no model selected."

    # Create a Gradio interface for PDF file upload
    iface = gr.Interface(
        fn=handle_upload,
        inputs=[
            gr.File(file_types=['.pdf']),
            gr.Dropdown(
                ["en_core_web_sm", "en_core_web_trf"],
                value="en_core_web_sm",
                label="Choose the model to be used."
            )
        ],
        outputs="text",
        live=False
    )

    # Launch the Gradio interface
    iface.launch(server_name='0.0.0.0', server_port=9000)

