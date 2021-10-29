import random

from nowcasting_dataset.filesystem.gcp import get_all_filenames_in_path, rename_file

# batch files are save with hash id and then batch and we want to remove the hashing
# {xxxxxx}_{batch_idx}.nc is the format of the file

train_path = "gs://solar-pv-nowcasting-data/prepared_ML_training_data/v5/train/"
validation_path = "gs://solar-pv-nowcasting-data/prepared_ML_training_data/v5/validation/"

train_filenames = get_all_filenames_in_path(remote_path=train_path)[1:]
validation_filenames = get_all_filenames_in_path(remote_path=validation_path)[1:]

random.shuffle(train_filenames)
random.shuffle(validation_filenames)

train_filenames = [file for file in train_filenames if "_" in file.split("/")[-1]]
validation_filenames = [file for file in validation_filenames if "_" in file.split("/")[-1]]


for filenames in [train_filenames, validation_filenames]:
    for file in train_filenames:

        print(file)

        filename = file.split("/")[-1]
        if "_" in filename:
            path = "/".join(file.split("/")[:-1]) + "/"
            new_filename = path + filename.split("_")[-1]

            try:
                rename_file(remote_file=file, new_filename=new_filename)
            except Exception as e:
                pass
        else:
            print(f"Skipping {filename}")
