import pandas as pd
from pathlib import Path
import slate3k as slate
import asyncio
import aiohttp
import time


loop = asyncio.get_event_loop()


async def get_pdf(url):
    async with aiohttp.ClientSession(loop=loop) as session:
        async with session.get(url) as response:
            return await response.content.read()


async def extract_data(pdf, name):
    try:
        with open(pdf, 'rb') as f:
            return slate.PDF(f)
    except Exception:
        print(f'Failed to get PDF content for {name}')
    return ''


async def send_request(url, name):
    extracted_pdf = ''
    try:
        filename = name.replace(' ', '_') + '.pdf'
        file = await get_pdf(url)
        pdf = Path(filename)
        pdf.write_bytes(file)
        extracted_pdf = await extract_data(pdf, name)
        pdf.unlink()  # Removes file to save storage
    except Exception:
        print(f'Failed to download data from {url}')

    return extracted_pdf


async def pdf_to_str(data):
    result = []
    for index, row in data.iterrows():
        name = row['name']
        link = row['link_white_paper']
        print(f'Sending request for {name}')
        start_time = time.perf_counter()
        result.append(await send_request(link, name))
        print(f'Successfully completed request for {name} in {time.perf_counter() - start_time}')
    data['Text'] = result
    return data


if __name__ == "__main__":
    start = time.perf_counter()
    df = pd.read_excel('/Users/ArnoldYanga/Downloads/ICOs-V1.xlsx')
    links = df[df['link_white_paper'].notnull()][['name', 'link_white_paper']]
    result = loop.run_until_complete(pdf_to_str(links))
    print(f'Completed entire process in {time.perf_counter() - start}')
    result.to_csv('/Users/ArnoldYanga/Desktop/Dataset/ICO3.csv')