from src.utils.prepare_mt_homograph_only_test import prepare_mt_homograph_only_test


def main():
    prepare_mt_homograph_only_test(
        lang1="en",
        lang2="de",
        min_count_per_language=3,
        input_dir="fairseq/examples/homograph_translation/orig/en_de_lowercase",
        output_dir="fairseq/examples/homograph_translation/orig/en_de_lowercase_homograph_only",
    )


if __name__ == "__main__":
    main()