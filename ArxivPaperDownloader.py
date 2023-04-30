import os
import re
import requests


class ArxivPaperDownloader:
    """
    A class to download arXiv papers based on their paper IDs.
    """

    def __init__(self, output_folder='arxiv_papers'):
        """
        Initializes the ArxivPaperDownloader with the specified output folder.
        
        :param output_folder: str, the folder where downloaded papers will be 
        saved (default: 'arxiv_papers')
        """
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def download_paper(self, arxiv_id):
        """
        Downloads an arXiv paper based on its ID and saves it in the specified output folder.

        :param arxiv_id: str, the paper ID to download
        """
        try:
            pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'
            response = requests.get(pdf_url, stream=True, timeout=10)
            if response.status_code == 200:
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

    def download_papers_from_file(self, filename):
        """
        Reads a text file containing paper IDs and titles, and downloads the corresponding papers.

        :param filename: str, the path to the text file
        """
        with open(filename, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            paper_ids = [
                self.extract_paper_id(line) for line in lines
                if self.extract_paper_id(line) is not None
            ]
            self.download_papers(paper_ids)

    @staticmethod
    def extract_paper_id(line):
        """
        Extracts the paper ID from a line of text.

        :param line: str, a line of text containing a paper ID
        :return: str or None, the extracted paper ID or None if not found
        """
        match = re.search(r'\[(\d{4}\.\d{5})\]', line)
        return match.group(1) if match else None


if __name__ == '__main__':
    FILENAME = 'paper_ids.txt'
    downloader = ArxivPaperDownloader('arxiv_papers')
    downloader.download_papers_from_file(FILENAME)
