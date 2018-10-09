import click

import plaintext_converter
import run
import json
import os
import shutil
import ntpath

from utilities import remove_doc_arrows

original_source_path = ''


@click.group()
def cli():
    pass


@cli.command()
@click.argument("source")
@click.argument("dest")
def ocr(source, dest):
    ocr_impl(source, dest)


def ocr_impl(source, dest):
    click.echo("Ocr")
    failed_files = plaintext_converter.convert_directory(source, dest)

    global original_source_path
    original_source_path = source
    remove_doc_arrows.clean_arrows(dest)
    failed_dir = 'FAILED_OCR_DIR'
    ensure_directory(failed_dir)
    for f in failed_files:
        filename = ntpath.basename(f)
        shutil.copyfile(f, failed_dir + '/' + filename)


@cli.command()
@click.argument("source")
@click.argument("dest")
def classify(source, dest):
    classify_impl(source, dest)


def classify_impl(source, dest):
        # TODO: put original files in new directory corresponding to class
    click.echo("Run Model")
    results = run.run_model(source)

    failed_source = 'FAILED_OCR_DIR'
    personal_dir = dest + '/PERSONAL'
    nonpersonal_dir = dest + '/NON_PERSONAL'
    sensitive_dir = dest + '/SENSITIVE_PERSONAL'
    ensure_directory(nonpersonal_dir)
    ensure_directory(personal_dir)
    ensure_directory(sensitive_dir)

    metadata_personal = open(personal_dir + '/metadata.json', 'a+')
    metadata_sensitive = open(sensitive_dir + '/metadata.json', 'a+')
    metadata_nonpersonal = open(nonpersonal_dir + '/metadata.json', 'a+')

    if os.path.isdir(failed_source):
        shutil.move(failed_source, dest)

    for path, category, individual_categories in results:
        print(f"File: {path} is {category}")
        filename = ntpath.basename(path)[:-4]
        original_filename = f"{original_source_path}/{filename}"
        print(f"Filename is {filename}")
        print(f"Source file is {original_filename}")

        output_file = ''
        #  import ipdb
        #  ipdb.set_trace()
        if category[0] == 0:
            output_file = nonpersonal_dir + '/' + filename
        if category[0] == 1:
            output_file = personal_dir + '/' + filename
            append_to_metadata_file(
                metadata_personal,
                filename,
                individual_categories
            )
        if category[0] == 2:
            output_file = sensitive_dir + '/' + filename
            append_to_metadata_file(
                metadata_sensitive,
                filename,
                individual_categories
            )


        #  import ipdb; ipdb.set_trace()
        print("Output File", output_file)
        shutil.copyfile(original_filename, output_file)
    metadata_personal.close()
    metadata_sensitive.close()
    metadata_nonpersonal.close()


@cli.command()
@click.argument("source")
@click.argument("dest")
def full(source, dest):
    validate_directories(source, dest)
    intermediate_directory = "intermediate_directory"
    ocr_impl(source, intermediate_directory)
    classify_impl(intermediate_directory, dest)
    shutil.rmtree(intermediate_directory)


def validate_directories(source, dest):
    if not os.path.isdir(source):
        print("Please give a directory as the first argument")
        return
    ensure_directory(dest)


def ensure_directory(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


def append_to_metadata_file(metadata_file, classified_file, categories):
    data = {classified_file: {'high_probability_categories': categories}}
    json.dump(data, metadata_file, sort_keys=True, indent=4)
    metadata_file.write('\n')


if __name__ == "__main__":
    cli()
