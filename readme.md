# A geographic agent-based approach to modeling healthcare accessibility

> &zwnj;**Note**&zwnj; (Last update: 2025.04.12):  
> The open-source implementation of our agent-based accessibility computation methodology is provided in this pre-release version to facilitate peer review and scholarly validation.  
> The visualization and comparative analysis modules are currently under active development.


## Dependencies
Ensure Python 3.8+ is installed, then install required packages:

Run `pip install -r requirements.txt`

## Usage

### Data Preparation
The repository includes sample datasets in the `sample_data` directory:
- `Health.csv`: Healthcare facility locations and attributes  
- `Area.csv`: Communitiy locations

To use your own data, please keep all original column names and data types and add new rows below the existing data in the CSV files  

### Execution
Run `python main.py`. Results will be generated in `results/` directory. 
