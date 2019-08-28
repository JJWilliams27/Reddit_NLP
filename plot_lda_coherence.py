# Plot Coherence for LDA Models

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

path = os.getcwd()
newpath = path +'/Outputs/CS_FULL/LDA_Dataframes/LDA_Model_Performance.csv'

lda_df = pd.read_csv(newpath)

ax = lda_df.plot(x='Num_Topics',y='Coherence',color='crimson',legend=False,linewidth=3)
ax.set_xlabel('Number of Topics',fontsize=18)
ax.set_ylabel('Coherence',fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=13)
#ax.grid(True, which='major', axis='x' )
ax.set_xticks(np.arange(2,32,2))
plt.show()