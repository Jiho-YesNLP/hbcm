"""
Install the package
python -m pip install azure-ai-textanalytics
"""

import code
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
import constants as cfg

credential = AzureKeyCredential(cfg.apikey)
ta_client = TextAnalyticsClient(endpoint=cfg.endpoint,
                                credential=credential)

doc = """The BQ and XBB subvariants of SARS-CoV-2 Omicron are now rapidly expanding, possibly due to altered antibody evasion properties deriving from their additional spike mutations. Here, we report that neutralization of BQ.1, BQ.1.1, XBB, and XBB.1 by sera from vaccinees and infected persons was markedly impaired, including sera from individuals boosted with a WA1/BA.5 bivalent mRNA vaccine. Titers against BQ and XBB subvariants were lower by 13- to 81-fold and 66- to 155-fold, respectively, far beyond what had been observed to date. Monoclonal antibodies capable of neutralizing the original Omicron variant were largely inactive against these new subvariants, and the responsible individual spike mutations were identified. These subvariants were found to have similar ACE2-binding affinities as their predecessors. Together, our findings indicate that BQ and XBB subvariants present serious threats to current COVID-19 vaccines, render inactive all authorized antibodies, and may have gained dominance in the population because of their advantage in evading antibodies."""

response = ta_client.extract_key_phrases([doc], language='en')

code.interact(local=dict(globals(), **locals()))

