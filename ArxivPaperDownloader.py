import os
import requests

class ArxivPaperDownloader:
    """
    Downloads arXiv papers based on their IDs and saves them in a specified output folder.
    """
    def __init__(self, output_folder='arxiv_papers'):
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def download_paper(self, arxiv_id):
        """
        Downloads an arXiv paper based on its ID and saves it in the specified output folder.

        :param arxiv_id: str, the paper ID to download
        """
        try:
            # Build the PDF URL based on the paper ID
            pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'

            # Send a GET request to the PDF URL
            response = requests.get(pdf_url, stream=True)

            # Check if the request was successful (status code: 200)
            if response.status_code == 200:
                # Save the PDF file in the output folder
                with open(os.path.join(self.output_folder, f'{arxiv_id}.pdf'), 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f'Successfully downloaded paper {arxiv_id} to {self.output_folder}')
            else:
                print(f'Failed to download paper {arxiv_id}: status code {response.status_code}')
        except Exception as exception:
            print(f'Error downloading paper {arxiv_id}: {exception}')

    def download_papers(self, paper_ids):
        """
        Downloads multiple arXiv papers based on their IDs.

        :param paper_ids: list of str, the paper IDs to download
        """
        for paper_id in paper_ids:
            self.download_paper(paper_id)


# Constructive feedback:
# 1. Create a class named ArxivPaperDownloader to better encapsulate functionality.
# 2. Separate the functionality of downloading a single paper and multiple papers into two different methods.
# 3. Use the 'stream=True' option in requests.get to handle large files more efficiently.
# 4. Use iter_content to download files in chunks.

# Example usage:

# Specify the paper IDs you want to download
paper_ids = [   '2303.08774', # GOT-4 Technical Report
                '2302.13971', # Llama Paper
                ""
            ]

# Instantiate the downloader with the output folder where you want to save the downloaded papers
downloader = ArxivPaperDownloader('arxiv_papers')

# Download the papers
downloader.download_papers(paper_ids)
