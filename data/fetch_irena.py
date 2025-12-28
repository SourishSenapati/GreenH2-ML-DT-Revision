import pandas as pd
import requests 

# Note: Since direct automated download might be blocked or require auth, 
# this script assumes you have manually downloaded the dataset if the URL fails.
# URL: https://www.irena.org/publications/2025/Green-Hydrogen

def fetch_data():
    try:
        # Placeholder for direct download if URL was direct CSV
        # response = requests.get('https://www.irena.org/.../data.csv')
        # with open('temp_data.xlsx', 'wb') as f: f.write(response.content)
        pass 
    except Exception as e:
        print(f"Auto-download failed: {e}. Please manually place 'IRENA_Green_H2_2025_Data.xlsx' in this folder.")

def process_data():
    try:
        # Assuming the file exists
        file_path = 'IRENA_Green_H2_2025_Data.xlsx'
        print(f"Processing {file_path}...")
        
        # Load data (adjust sheet_name as per actual file)
        # Using 'Sheet1' as default fallback if 'Electrolysis_Logs' doesn't exist
        try:
            df = pd.read_excel(file_path, sheet_name='Electrolysis_Logs')
        except:
            df = pd.read_excel(file_path)
            
        print("Data loaded successfully.")
        print(df.head())
        
        # Save to CSV for the simulation to use
        output_file = 'blended_data.csv'
        df.to_csv(output_file, index=False)
        print(f"Saved processed data to {output_file}")
        
    except FileNotFoundError:
        print("Error: 'IRENA_Green_H2_2025_Data.xlsx' not found. Please download it from IRENA.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    process_data()
