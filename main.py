from data import load_data
from word_embedding import create_word_doc_freq

def main():
    train_path = "./learn-ai-bbc/BBC_News_Train.csv"
    test_path = "./learn-ai-bbc/BBC_News_Test.csv"
    sample_solution_path = "./learn-ai-bbc/BBC_News_Sample_Solution.csv"

    train_df, test_df, sample_solution_df = load_data(
        train_path, test_path, sample_solution_path
    )

    wordtermfreq_df = create_word_doc_freq(train_df, "Text")
    

if __name__ == "__main__":
    main()
