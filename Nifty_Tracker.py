import requests
import pandas as pd
from datetime import datetime, timedelta
import configparser
import json
import os
import time
from openai import OpenAI
import pdfplumber
import re
import dotenv
from tqdm import tqdm
import pymupdf4llm
dotenv.load_dotenv()

class Tracker:
    def __init__(self, config_path: str):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.data = []
        self.fetched = set()
        self.processed_ids = set()
        self.processed_date = None
        self.data = []

    def _processed_path(self, date_str: str):
        return f"{date_str}_processed.json"

    def load_processed_for_date(self, date_str: str):
        self.processed_date = date_str
        path = self._processed_path(date_str)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    ids = json.load(f)
                if isinstance(ids, list):
                    self.processed_ids = set(ids)
                else:
                    self.processed_ids = set()
            except Exception:
                self.processed_ids = set()
        else:
            self.processed_ids = set()

    def save_processed(self):
        if not self.processed_date:
            return
        path = self._processed_path(self.processed_date)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(sorted(list(self.processed_ids)), f)
        except Exception:
            pass

    def getAnnouncements(self, code: int, prevDate: str = None, toDate: str = None, test: bool = False, **kwargs):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.bseindia.com/",
            "Origin": "https://www.bseindia.com",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9"
        }
        session = requests.Session()
        session.headers.update(headers)
        
        if toDate is None:
            to_date_obj = datetime.now()
            strToDate_val = to_date_obj.strftime("%Y%m%d")
        else:
            try:
                to_date_obj = datetime.strptime(toDate, "%Y%m%d")
                strToDate_val = toDate
            except ValueError:
                print("Error parsing dates, defaulting to today")
                to_date_obj = datetime.now()
                strToDate_val = to_date_obj.strftime("%Y%m%d")

        if prevDate is None:
            prev_date_obj = to_date_obj - timedelta(days=1)
            strPrevDate_val = prev_date_obj.strftime("%Y%m%d")
        else:
            prevDate_str = str(prevDate)
            try:
                prev_date_obj = datetime.strptime(prevDate_str, "%Y%m%d")
                strPrevDate_val = prevDate_str
            except ValueError:
                prev_date_obj = to_date_obj - timedelta(days=1)
                strPrevDate_val = prev_date_obj.strftime("%Y%m%d")

        cat = kwargs.get('cat', 'Result')
        subCat = kwargs.get('subCat','Financial Results')

        params = {
            "pageno": 1,
            "strCat": cat,
            "strPrevDate": strPrevDate_val,
            "strScrip": str(code),
            "strSearch": "P",
            "strToDate": strToDate_val,
            "strType": "C",
            "subcategory": subCat
        }

        if test:
            params.pop('strScrip')
        session.params.update(params)
        response = session.get(url=self.config['URLs']['Feed'])
        try:
            self.data.extend(response.json()['Table'])
            return response.json()
        except Exception as e:
            print(e)
            return None
    
    def getProcessed(self):
        message = ""
        _data = []
        for r in self.data:
            news = r['NEWSSUB']
            fileurl = f"{self.config['URLs']['Doc']}?Pname={r['ATTACHMENTNAME']}"
            pub_time = r['DT_TM']
            id = r['NEWSID']
            name = r['SLONGNAME']
            _data.append({
                'id': id,
                'company': name,
                'sec_code': r['SCRIP_CD'],
                'headline': news,
                'file_url': fileurl,
                'time': pub_time
            })
            message += f"🏢 {name}\n📢 {news}\n⏰ {pub_time}\n🔗 {fileurl}\n\n"
        return message, pd.DataFrame(_data)
    
    def format_date(self, date: datetime):
        date_str = str(date)
        date_str = date_str.split(" ")[0].replace("-", "")
        return date_str
    
    def build_set(self, df: pd.DataFrame, prevDate: str = None, toDate: str = None):
        # Reset accumulated announcements before a new polling batch
        for i, row in tqdm(df.iterrows(), desc='Processing Announcements'):
            self.getAnnouncements(code=row['Security Code'], prevDate=prevDate, toDate=toDate)
            time.sleep(0.2)

class Bot:
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.client = OpenAI()
        with open("data/subscribers.json", "r") as f:
            self.mailing_list = set(json.loads(f.read().strip()))
        self.min_delay_between_calls = float(os.getenv("OPENAI_MIN_DELAY", "0.4"))
        self._last_openai_call_ts = 0.0

        ## Keywords to identify relevant financial pages
        self.page_filter = re.compile(
            r"("
            r"financial\s+highlights|"
            r"results?|quarterly\s+performance|summary\s+of\s+results?|"
            r"profit\s*(after|before)?\s*(tax|provisions)?|"
            r"revenue|income\s+from\s+operations|total\s+income|"
            r"interest\s+(income|earned)|finance\s+costs?|"
            r"expenses?|operating\s+expenditure|opex|"
            r"ebitda|ebit|pbt|pat|net\s+profit|loss|"
            r"earnings\s+per\s+share|eps|"
            r"loans?\s+(and\s+advances|book|portfolio|outstanding)|"
            r"advances?|aum|assets?\s+under\s+management|"
            r"borrowings?|deposits?|capital\s+adequacy|car\b|"
            r"tier\s*1|tier\s*i|crar|"
            r"provisions?|expected\s+credit\s+loss|ecl|impairment|"
            r"gross\s+npas?|net\s+npas?|gnpa|nnpa|"
            r"disbursements?|collections?|liabilities?|balance\s+sheet|"
            r"ratios?|financial\s+ratios?|"
            r"employee\s+benefits?|staff\s+cost|personnel\s+expenses|"
            r"other\s+expenses|overheads?"
            r")",
            re.IGNORECASE
        )


    # --- TEXT EXTRACTION ---
    def extract_pages(self, pdf_path):
        """Extract all text from PDF pages using pdfplumber."""
        pages_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                # Skip empty or useless pages
                if text.strip():
                    pages_text.append((i, text.strip()))
        return pages_text

    def extract_to_md(self, pdf_path):
        md_text = pymupdf4llm.to_markdown(pdf_path)
        return md_text
    
    def filter_financial_pages(self, pages_text):
        """Filter out pages unrelated to financial results."""
        return [(i, text) for i, text in pages_text if self.page_filter.search(text)]

    # --- AI EXTRACTION ---
    def extract_with_ai(self, filtered_pages):
        """Extract numeric KPIs from filtered text using GPT."""
        if not filtered_pages:
            return None

        # # Combine relevant text (limit to avoid context overflow)
        # text_content = "\n\n".join(
        #     [f"Page {i + 1}:\n{text}" for i, text in filtered_pages[:6]]
        # )[:20000]

        text_content = filtered_pages

        system_msg = (
            "You are a financial data extraction assistant. "
            "Your task is to read text excerpts from quarterly financial reports "
            "and extract or compute numerical KPIs accurately. "
            "Return only a single valid JSON object with the requested fields. No prose, no code, no markdown."
        )

        user_msg = (
            "You are reading a quarterly financial report of an NBFC or financial institution.\n"
            "Extract **numerical values only** (no units, %, ₹, or Cr) for the following keys "
            "from the most recent quarter in the text:\n\n"
            "1. 'Interest Income'\n"
            "2. 'Finance Costs'\n"
            "3. 'Provisions / ECL / Loan Losses'\n"
            "4. 'Employee Benefits Expense'\n"
            "5. 'Other Expenses'\n"
            "6. 'EPS Basic'\n"
            "7. 'EPS Diluted'\n"
            "8. 'Loans to Customers'\n"
            "9. 'CAR (%)'\n"
            "10. 'Tier 1 Ratio (%)'\n\n"
            "Rules:\n"
            "- If not present, set value to empty string.\n"
            "- Use only the **latest quarter**, not YTD or full year.\n"
            "- Return strictly JSON with only those keys.\n"
            "- Example:\n"
            "{\n"
            '  "Interest Income": "12345",\n'
            '  "Finance Costs": "6789",\n'
            '  "Provisions / ECL / Loan Losses": "123",\n'
            '  "Employee Benefits Expense": "456",\n'
            '  "Other Expenses": "789",\n'
            '  "EPS Basic": "12.34",\n'
            '  "EPS Diluted": "12.10",\n'
            '  "Loans to Customers": "567890",\n'
            '  "CAR (%)": "18.5",\n'
            '  "Tier 1 Ratio (%)": "15.2"\n'
            "}\n\n"
            f"Text to extract from:\n{text_content}"
        )

        # --- Retry logic ---
        max_retries = 5
        backoff = 2.0
        for attempt in range(max_retries):
            now = time.time()
            delay = self.min_delay_between_calls - (now - self._last_openai_call_ts)
            if delay > 0:
                time.sleep(delay)
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4.1",  # use gpt-4o-mini or gpt-4.1 depending on access
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    timeout=90,
                )
                self._last_openai_call_ts = time.time()
                return resp.choices[0].message.content
            except Exception as e:
                err_text = str(e).lower()
                if any(x in err_text for x in ["rate", "429", "timeout", "unavailable"]):
                    if attempt < max_retries - 1:
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 60)
                        continue
                print(f"❌ Extraction failed: {e}")
                return None

    # --- PIPELINE ---
    def buildNumbers(self, data: pd.DataFrame):
        results = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing PDFs"):
            pdf_path = os.path.join("PDFs", row["file_url"].split("=")[1])
            # pages_text = self.extract_pages(pdf_path)
            # filtered = self.filter_financial_pages(pages_text)
            md_text = self.extract_to_md(pdf_path)
            ai_result = self.extract_with_ai(md_text)

            if not ai_result:
                continue

            try:
                result = json.loads(ai_result)
            except Exception:
                # Handle malformed JSON by regex extraction
                match = re.findall(r"\{[\s\S]*\}", ai_result)
                if not match:
                    print("Invalid JSON from model:", ai_result)
                    continue
                result = json.loads(match[-1])

            result.update({
                "company": row["company"],
                "date": row["time"],
                "sec_code": row["sec_code"],
                "file_url": row["file_url"],
            })
            results.append(result)

        # Build final dataframe
        cols = [
            "company", "date", "sec_code",
            "Interest Income", "Finance Costs", "Provisions / ECL / Loan Losses",
            "Employee Benefits Expense", "Other Expenses",
            "EPS Basic", "EPS Diluted", "Loans to Customers",
            "CAR (%)", "Tier 1 Ratio (%)", "file_url"
        ]
        return pd.DataFrame(results, columns=cols)

    # --- TELEGRAM SENDER ---
    def send_messages(self, message, data_path):
        BOT_TOKEN = self.config["Bot"]["Token"]
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
        with open(data_path, "rb") as f:
            for chat_id in self.mailing_list:
                f.seek(0)
                requests.post(
                    url,
                    data={
                        "chat_id": chat_id,
                        "caption": message,
                        "parse_mode": "Markdown",
                    },
                    files={"document": (os.path.basename(data_path), f, "text/csv")},
                )

def getReport(url, quarter=None):
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    "Referer": "https://www.bseindia.com/",
    "Origin": "https://www.bseindia.com",
    }
    os.makedirs("PDFs", exist_ok=True)
    res = requests.get(url, headers=headers, timeout=20)

    if res.status_code == 200:
        path = os.path.join("PDFs", url.split('=')[1])
        with open(path, "wb") as f:
            f.write(res.content)

def fetch_historical_quarterly():
    os.makedirs('NBFC_QuarterlyResults', exist_ok=True)
    df = pd.read_excel("RequiredNBFCs.xlsx")
    # Last 10 quarter end dates (most recent first, ending 30-Jun-2025)
    quarter_ends = [
        "2025-06-30",
        "2025-03-31",
        "2024-12-31",
        "2024-09-30",
        "2024-06-30",
        "2024-03-31",
        "2023-12-31",
        "2023-09-30",
        "2023-06-30",
        "2023-03-31",
        "2022-12-31",
        "2022-09-30",
        "2022-06-30",
        "2022-03-31",
        "2021-12-31",
        "2021-09-30",
        "2021-06-30",
        "2021-03-31",
        "2020-12-31",
        "2020-09-30",
        "2020-06-30",
        "2020-03-31"
    ]

    # Convert to datetime
    quarter_ends = [datetime.strptime(d, "%Y-%m-%d") for d in quarter_ends]

    # Compute upload windows (45 days after quarter end)
    upload_windows = []
    for q_end in quarter_ends:
        tracker = Tracker(config_path='.ini')
        bot = Bot(config_path='.ini')
        start = q_end + timedelta(days=1)
        end = start + timedelta(days=30)
        start1 = end + timedelta(days=1)
        end1 = start1 + timedelta(days=14)   # 45 days inclusive
        
        prev = tracker.format_date(start)
        to = tracker.format_date(end)
        prev1 = tracker.format_date(start1)
        to1 = tracker.format_date(end1)

        tracker.build_set(df, prev, to)
        tracker.build_set(df, prev1, to1)

        _, fetched = tracker.getProcessed()
        fetched = fetched.drop_duplicates(subset=['company'])
        fetched.to_csv(os.path.join('NBFC_QuarterlyResults', f"{tracker.format_date(q_end)}_announcements.csv"), index=False)
        if not fetched.empty:
            # for _, row in fetched.iterrows():
            #     url = row['file_url']
            #     getReport(url, tracker.format_date(q_end))

            extracted = bot.buildNumbers(fetched)
            extracted.to_csv(os.path.join('NBFC_QuarterlyResults', f"{tracker.format_date(q_end)}_numbers.csv"), index=False)


def main():
    tracker = Tracker(config_path='.ini')
    df = pd.read_csv("tracking_list.csv")
    bot = Bot(config_path='.ini')
    _today = tracker.format_date(datetime.today())
    tracker.load_processed_for_date(_today)
    while True:
        save_path = f"{_today}.csv"
        today = tracker.format_date(datetime.today())
        prev = tracker.format_date(datetime.today() - timedelta(days=0))
        tracker.build_set(df, prev, today) # For deployment

        _, data = tracker.getProcessed()

        # Filter out announcements already processed today
        if not data.empty:
            new_data = data[~data['id'].isin(tracker.processed_ids)].copy()
        else:
            new_data = data
        if not new_data.empty:
            # Download only new PDFs
            for _, row in new_data.iterrows():
                url = row['file_url']
                getReport(url)

            # Run extraction only on new announcements
            extracted_data = bot.buildNumbers(data=new_data)
            extracted_data.to_csv(save_path, index=False)

            # # Build message only for new announcements
            # new_message = ""
            # for _, row in new_data.iterrows():
            #     new_message += f"🏢 {row['company']}\n📢 {row['headline']}\n⏰ {row['time']}\n🔗 {row['file_url']}\n\n"

            # if new_message:
            #     bot.send_messages(new_message, save_path)

            # Mark as processed for today and persist
            tracker.processed_ids.update(new_data['id'].tolist())
            tracker.save_processed()

        time.sleep(60)

        if today != _today:
            _today = tracker.format_date(datetime.today())
            tracker.fetched = set()
            tracker.load_processed_for_date(_today)

        break
if __name__ == '__main__':
    fetch_historical_quarterly()