# rcs_cataloging
For scraping RCS JSON files and adding to centralized database.


Simple rules used to generate the database:

- If a the millivolt values of all channels in a time domain packet are identical, the packet is discarded.