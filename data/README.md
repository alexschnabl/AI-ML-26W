# Data

The raw data is **not** committed to this repository. It is subject to the
[competition rules](https://www.kaggle.com/competitions/home-credit-default-risk/rules)
of the Kaggle competition [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk).

## Download

1. Sign in to Kaggle and accept the rules on the competition page.
2. Create an API token under *Settings → API → Create New Token* and place `kaggle.json` in
   `~/.kaggle/` (Windows: `%USERPROFILE%\.kaggle\`).
3. From the repository root, run:

```bash
kaggle competitions download -c home-credit-default-risk -p data/raw
unzip data/raw/home-credit-default-risk.zip -d data/raw
```

## Expected files (`data/raw/`)

| File | Content |
| --- | --- |
| `application_train.csv` | Main table: about 307,500 labeled applications (`TARGET`) |
| `application_test.csv` | About 48,700 applications without labels, for Kaggle submissions |
| `bureau.csv`, `bureau_balance.csv` | Credit history at other institutions (credit bureau) |
| `previous_application.csv` | Previous Home Credit applications |
| `POS_CASH_balance.csv`, `credit_card_balance.csv`, `installments_payments.csv` | Monthly balances and repayment history |
| `HomeCredit_columns_description.csv` | Column descriptions |
| `sample_submission.csv` | Submission format |

Processed or intermediate files go into `data/processed/`, which is ignored by git as well.
