"""
Builds data/test_dataset.csv - the labelled test data set used to test SMSA.

60 financial texts = 5 text types x 12 texts, balanced 4 Positive / 4 Negative /
4 Neutral per type. Company names are fictional so that no real company is
associated with invented news. The expected label was assigned manually by the
team using the label definitions in the report (impact on the company's market
sentiment from an investor's point of view).
"""
import csv
import os

P, N, U = "Positive", "Negative", "Neutral"

DATA = {
    # ------------------------------------------------------------------ Financial news
    "news": [
        ("Aurora Steel Ltd", P, "Aurora Steel Ltd reported a 32% year-on-year jump in quarterly net profit to Rs 1,845 crore, beating street estimates, as higher realisations and lower coking coal costs lifted margins. The stock rose 6% in early trade."),
        ("Brightwave Technologies", P, "Brightwave Technologies shares hit a 52-week high after the IT services firm signed a five-year, $750 million digital transformation deal with a leading US bank, its largest contract to date."),
        ("Sunrise Pharma", P, "Sunrise Pharma received final approval from the US FDA for its generic version of a blockbuster diabetes drug, opening a market estimated at $1.2 billion annually. Analysts expect the launch to add meaningfully to earnings from next year."),
        ("Kaveri Power Ltd", P, "Kaveri Power's net loss narrowed sharply to Rs 45 crore from Rs 390 crore a year ago, far better than analysts had feared, as plant utilisation improved. Shares surged 9%."),
        ("Pinnacle Motors", N, "Pinnacle Motors shares slumped 11% after the automaker reported a 40% drop in quarterly profit and warned that a semiconductor shortage would hit production for the rest of the year."),
        ("Everest Bank", N, "The central bank imposed a penalty of Rs 25 crore on Everest Bank for lapses in KYC compliance and barred it from onboarding new credit-card customers until the deficiencies are fixed."),
        ("Lotus Retail", N, "Lotus Retail's quarterly revenue grew 8%, but net profit missed consensus estimates by a wide margin as discounting and higher rent costs squeezed margins; the stock fell 5% on the results."),
        ("Greenfield Agro", N, "Rating agency CRISTAL downgraded Greenfield Agro's long-term debt to 'D' after the company delayed interest payments on its non-convertible debentures, citing a severe liquidity crunch."),
        ("Coastal Cements", U, "Coastal Cements will announce its results for the quarter ended September on October 24, the company said in a regulatory filing on Monday."),
        ("Meridian Finance", U, "Meridian Finance's quarterly results were broadly in line with expectations: loan growth of 14% was offset by a slight rise in credit costs, and the stock ended the day flat."),
        ("Horizon Airlines", U, "Horizon Airlines said it will shift its corporate headquarters from Gurugram to a new office in Delhi's Aerocity by March next year."),
        ("Nexus Telecom", U, "Nexus Telecom's shares will be included in the Nifty Midcap 150 index and excluded from the Nifty Smallcap 250 index as part of the semi-annual index rebalancing effective from next month."),
    ],
    # ------------------------------------------------------------------ Company announcements
    "announcement": [
        ("Aurora Steel Ltd", P, "Aurora Steel Ltd is pleased to announce that the Board of Directors has approved a buyback of equity shares worth Rs 2,000 crore at a price of Rs 1,450 per share, a premium of 22% to the current market price."),
        ("Sunrise Pharma", P, "Sunrise Pharma announces that the Board has recommended a final dividend of Rs 18 per share, the highest in the company's history, in addition to the interim dividend of Rs 7 paid earlier in the year."),
        ("Vertex Infra Projects", P, "Vertex Infra Projects Ltd hereby informs that it has been declared the lowest bidder (L1) for a Rs 3,400 crore metro rail project, taking the company's order book to a record Rs 28,000 crore."),
        ("Brightwave Technologies", P, "Brightwave Technologies announces the successful completion of the acquisition of CloudNine Analytics, which is expected to be earnings-accretive from the first full year and adds 120 enterprise clients."),
        ("Greenfield Agro", N, "Greenfield Agro Ltd informs the exchange that it has defaulted on the repayment of principal and interest of Rs 310 crore due to its lenders on 30 September."),
        ("Pinnacle Motors", N, "Pinnacle Motors announces the temporary suspension of operations at its Pune manufacturing plant with immediate effect following a major fire; the plant accounts for nearly 35% of the company's output."),
        ("Lotus Retail", N, "Lotus Retail Ltd informs that its Chief Financial Officer and the statutory auditor have resigned with immediate effect, and the auditor has raised concerns about certain related-party transactions."),
        ("Everest Bank", N, "Everest Bank announces that the Enforcement Directorate has initiated an investigation into alleged irregularities in loans sanctioned to a group of real-estate companies, and has attached assets worth Rs 800 crore."),
        ("Coastal Cements", U, "Coastal Cements Ltd hereby informs that a meeting of the Board of Directors is scheduled on 24 October to consider and approve the unaudited financial results for the quarter ended 30 September."),
        ("Meridian Finance", U, "Pursuant to Regulation 30 of the SEBI (LODR) Regulations, Meridian Finance Ltd informs that the trading window for dealing in its securities will remain closed for designated persons until 48 hours after the declaration of results."),
        ("Horizon Airlines", U, "Horizon Airlines Ltd informs that the 18th Annual General Meeting of the company will be held on 12 August through video conferencing, and the record date for the purpose of the AGM is 5 August."),
        ("Nexus Telecom", U, "Nexus Telecom Ltd announces the appointment of M/s Rao & Associates, Chartered Accountants, as the secretarial auditor of the company for a term of five years, subject to shareholders' approval."),
    ],
    # ------------------------------------------------------------------ Analyst commentary
    "analyst": [
        ("Brightwave Technologies", P, "We upgrade Brightwave Technologies to BUY from HOLD and raise our target price to Rs 1,850, implying 28% upside. Strong deal wins and improving margins give us confidence in double-digit earnings growth over FY26-28."),
        ("Aurora Steel Ltd", P, "Aurora Steel remains our top pick in the metals space. Capacity expansion is on track, costs are falling and we see return on equity improving to 20% by FY27. Reiterate OUTPERFORM."),
        ("Kaveri Power Ltd", P, "Kaveri Power's turnaround is gaining momentum: we expect the company to turn profitable next quarter as the new tariff order kicks in. We initiate coverage with a BUY rating and a target of Rs 260."),
        ("Sunrise Pharma", P, "Sunrise Pharma's US pipeline is the strongest among Indian peers, with 12 approvals expected over two years. We raise our FY27 EPS estimate by 15% and maintain ADD with a higher target price."),
        ("Pinnacle Motors", N, "We downgrade Pinnacle Motors to SELL. Rising competition in the EV segment, market-share losses and elevated inventory levels are likely to pressure margins; we cut our target price by 25% to Rs 540."),
        ("Lotus Retail", N, "Lotus Retail's same-store sales growth has decelerated for the fourth consecutive quarter. With store expansion slowing and rentals rising, we see downside risk to consensus earnings and maintain UNDERPERFORM."),
        ("Greenfield Agro", N, "Given the default on its debt and the absence of a credible refinancing plan, we suspend our rating on Greenfield Agro and advise investors to exit; equity holders face a meaningful risk of dilution or loss."),
        ("Everest Bank", N, "Everest Bank's asset-quality deterioration is worse than we expected: gross NPAs rose to 6.8% and slippages in the unsecured book remain high. We cut our earnings estimates by 18% and move to REDUCE."),
        ("Meridian Finance", U, "We maintain our HOLD rating on Meridian Finance with an unchanged target price. Loan growth and margins are tracking our estimates, and the current valuation fairly reflects the company's prospects."),
        ("Coastal Cements", U, "Coastal Cements' results were in line with our forecasts. Volume growth was healthy, but pricing remained weak; we see balanced risks and keep our NEUTRAL stance."),
        ("Horizon Airlines", U, "Horizon Airlines benefits from strong travel demand, but high fuel prices and a stretched balance sheet offset the positives. We initiate coverage with an EQUAL-WEIGHT rating."),
        ("Nexus Telecom", U, "Nexus Telecom's tariff hike should support revenue, but the higher spectrum payments due next year will absorb most of the benefit. Our estimates are unchanged and we reiterate HOLD."),
    ],
    # ------------------------------------------------------------------ Investor comments
    "investor": [
        ("Sunrise Pharma", P, "I've held Sunrise Pharma since 2019 and the management has delivered on every promise. With the new FDA approvals, I'm increasing my position - this is a long-term compounder."),
        ("Aurora Steel Ltd", P, "Aurora Steel's buyback at a 22% premium shows the promoters believe the stock is undervalued. As a shareholder I'm very happy and will tender part of my holding."),
        ("Vertex Infra Projects", P, "Vertex Infra's order book is now almost five times its annual revenue. Execution has been solid, so I expect strong earnings visibility for years. Adding more to my portfolio."),
        ("Kaveri Power Ltd", P, "Finally some good news for Kaveri Power shareholders - losses are shrinking every quarter and the debt is coming down. I think the worst is behind us."),
        ("Lotus Retail", N, "Sold my entire stake in Lotus Retail today. The auditor resigning and raising red flags about related-party deals is a deal-breaker for me. Corporate governance matters."),
        ("Greenfield Agro", N, "I'm an investor in Greenfield Agro and I'm deeply disappointed. The company kept assuring us that there was no liquidity problem and now it has defaulted. My investment is down 70%."),
        ("Pinnacle Motors", N, "Pinnacle Motors keeps losing market share to newer EV players and management has no clear answer. I don't see any reason to hold this stock anymore."),
        ("Everest Bank", N, "The ED investigation into Everest Bank's loans worries me a lot. Until there is clarity, I'm reducing my exposure - there could be more skeletons in the closet."),
        ("Meridian Finance", U, "I hold some Meridian Finance shares. Does anyone know when the record date for the interim dividend is?"),
        ("Coastal Cements", U, "Coastal Cements looks fairly valued to me at current levels. I'll keep my existing holding but won't add more until I see the next set of results."),
        ("Horizon Airlines", U, "Horizon Airlines has good passenger growth but carries a lot of debt. The positives and negatives balance out for me, so I'm staying on the sidelines for now."),
        ("Nexus Telecom", U, "Nexus Telecom is moving to the Midcap index next month. Not sure whether that will change anything for long-term investors like me."),
    ],
    # ------------------------------------------------------------------ Social-media-style comments
    "social": [
        ("Brightwave Technologies", P, "$BRWV just smashed earnings and landed a $750M deal 🚀🚀 This one is going to the moon! Loading up more tomorrow #bullish"),
        ("Vertex Infra Projects", P, "Vertex Infra bagging order after order 🔥 L1 in a Rs 3,400 cr metro project. Infra theme is on fire and $VRTX is leading it. Super bullish!"),
        ("Sunrise Pharma", P, "Sunrise Pharma record dividend + FDA nod in the same month?? Dream run for shareholders 💰📈 Holding tight"),
        ("Kaveri Power Ltd", P, "Kaveri Power up 9% today! The turnaround story is real. Bought the dip last month, feeling great 😎 #KaveriPower"),
        ("Greenfield Agro", N, "Greenfield Agro defaulted again 🤡 total rug pull. Anyone still holding this is a bagholder. Stay far away!!"),
        ("Lotus Retail", N, "Auditor quits, CFO quits... wow, Lotus Retail is really setting new standards in corporate governance 👏👏 Can't wait to see what else they're hiding lol"),
        ("Pinnacle Motors", N, "$PNCL down 11% after that profit crash 📉📉 Plant fire, chip shortage, losing EV market share... this stock is a disaster. Dumping everything."),
        ("Everest Bank", N, "ED raids at Everest Bank 😬 penalty last month, investigation this month. Something is seriously wrong there. Exit before it gets worse. #EverestBank"),
        ("Coastal Cements", U, "Coastal Cements results are out on Oct 24. What are you all expecting? Drop your estimates below 👇"),
        ("Meridian Finance", U, "Meridian Finance closed flat today after results. Nothing exciting either way, just in line. #stockmarket"),
        ("Horizon Airlines", U, "Just saw that Horizon Airlines is moving its HQ to Aerocity. Interesting move. Anyone know why?"),
        ("Nexus Telecom", U, "Index rebalancing update: Nexus Telecom moves from Smallcap 250 to Midcap 150 from next month. FYI for index fund followers."),
    ],
}


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_dataset.csv")
    rows, tid = [], 1
    for text_type, items in DATA.items():
        assert len(items) == 12, text_type
        for company, label, text in items:
            rows.append({"test_id": tid, "text_type": text_type, "company_name": company,
                         "input_text": text, "expected_sentiment": label})
            tid += 1
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} texts to {out}")


if __name__ == "__main__":
    main()
