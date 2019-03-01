## SEC 13F Scraper

The SEC 13F scraper will be used to understand the holdings of mutual funds. The base of this code is borrowed from some tips provided by [Brian Caffey](http://briancaffey.github.io/2018/01/30/reading-13f-sec-filings-with-python.html) and [Christian Packard](https://github.com/cpackard/fundholdings).
The goal here is to create a library of useful functions and maybe classes to be used in a rolling analysis of US mutual funds.

## SEC EDGAR

Get the holdings info from EDGAR, a tool for looking up holdings disclosures for mutual funds, ETFs, etc. Guide to using EDGAR is [here](https://www.sec.gov/oiea/Article/edgarguide.html). Search for US Mutual Funds [here](https://www.sec.gov/edgar/searchedgar/mutualsearch.html).

### What files do we want?

* **N-Q:** Quarterly schedule of portfolio holdings – Includes a list of the fund’s portfolio holdings for the first and third fiscal quarters, those not reported on Form N-CSR.
* **N-CSR/N-CSRS:** Annual/semi-annual shareholder reports – Describes how the fund has operated and includes the fund’s holdings and financial statements.  The annual report also discusses market conditions and investment strategies that significantly affected the fund’s performance during its last fiscal year.
