from nowcasting_dataset.cloud.gcp import get_all_filenames_in_path, rename_file



# batch files are save with hash id and then batch and we want to remove the hashing
# {xxxxxx}_{batch_idx}.nc is the format of the file

train_path = 'gs://solar-pv-nowcasting-data/prepared_ML_training_data/v5/train/'
validation_path = 'gs://solar-pv-nowcasting-data/prepared_ML_training_data/v5/validation/'

train_filenames = get_all_filenames_in_path(remote_path=train_path)
validation_filenames = get_all_filenames_in_path(remote_path=validation_path)

for filenames in [train_filenames, validation_filenames]:
    for file in train_filenames[1:]:
        path = train_filenames[0]

        filename = file.split('/')[-1]
        new_filename = path + filename[7:]

        print(new_filename)

        rename_file(remote_file=file, new_filename=new_filename)