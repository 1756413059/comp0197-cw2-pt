# OEQ (Open-Ended Question)

**To run the ViT Segmentaion Part, open your terminal and type:**

python main.py

This command will do the following:

**Download and Preprocess Data:**

- It downloads the OxfordIIITPet dataset and preprocesses it.

**Generate Pseudo Labels:**

- It runs a function to generate pseudo labels using a ViT model. The images and masks are saved in the output_vit/images and output_vit/masks folders.

**Train the U-Net Model:**

- It trains a U-Net model using the pseudo labeled data. The model is saved in the saved_models folder.

**Evaluate the Model:**

- It evaluates the model on the test set and shows the IoU and Dice scores.

Simply run python main.py to execute the whole process.
