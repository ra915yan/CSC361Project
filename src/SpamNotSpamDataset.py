import config
import random
import pandas as pd




import config
import random
import pandas as pd

class SpamNotSpamDataset:
    def __init__(self, file_path=None):
        if file_path is None:
            file_path = config.DATA1_PATH
            
        self.df = pd.read_csv(file_path)
        if self.df.empty:
            raise ValueError("Dataset is empty.")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        return row['email'], row['label']
    
    def get_random_sample(self):
        """Standardized to lowercase 's' to match GUI calls."""
        return self.df.sample(n=1).iloc[0]