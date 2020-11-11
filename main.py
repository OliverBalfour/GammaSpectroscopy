
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def parse_spe (fname):
  """Parses a .spe file to dict of counts, time, channel bounds."""
  with open(fname, 'r') as f:
    lines = f.read().split('\n')
  time_line = lines[lines.index("$MEAS_TIM:") + 1].split(' ')
  alive_time, total_time = [int(time_line[0]), int(time_line[1])]
  channels_start = lines.index("$DATA:") + 2
  channel_bounds = lines[channels_start - 1].split(' ')
  min_channel, max_channel = [int(channel_bounds[0]), int(channel_bounds[1])]
  num_channels = max_channel - min_channel
  counts = [int(x) for x in lines[channels_start:channels_start+num_channels+1]]
  spe = { "counts": np.array(counts), "channels": [min_channel, max_channel],
    "alive_time": alive_time, "total_time": total_time }
  return spe

def normalise_counts (spe):
  """Normalise counts to be per unit (alive) time. Allows comparison."""
  return { **spe, "counts": spe["counts"] / spe["alive_time"] }

background = normalise_counts(parse_spe("Background.spe"))
def subtract_background_and_normalise ():
  """Subtract normalised background counts and normalise to cnt/unit time."""
  def inner(spe):
    new_spe = normalise_counts(spe)
    return { **spe, "counts": new_spe["counts"] - background["counts"] }
  return inner

def plot_channel (spe, fact=1):
  # +1 due to exclusivity; need dimensions to align
  plt.plot(np.arange(spe["channels"][0],spe["channels"][1]+1), spe["counts"]*fact)

def plot_titles (title="", xlabel="", ylabel=""):
  plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)

def plot_peaks (spes, energy_levels):
  plt.yscale("log")
  for energy_level in energy_levels:
    plt.scatter(np.arange(len(spes)), [spe["peak"][energy_level][1][0] for spe in spes], label=energy_level)
  plt.legend(loc=9, prop={'size':6})
  plt.show()

def peak_channel (spe, minch, maxch):
  """
  Find exact peak between min and max channel if background subtracted.
  Peak = \sum_{i=a}^b \frac{x_i n_i}{n_i}
  """
  # TODO: np.dot(np.arange, counts[minch:maxch])
  # TODO: account for Compton background
  return (sum([i * spe["counts"][i] for i in range(minch,maxch+1)])
        / sum([    spe["counts"][i] for i in range(minch,maxch+1)]))

def peak_area (spe, a, b):
  """
  Compute area under count array in interval [a,b], subtracting the Compton background.
  Area = \sum_{i=a}^b Gross_i - \frac{Gross_a + Gross_b}{2} \mul (b - a)

  Uncertainty = \sqrt((\Delta Compton)^2 + (\Delta Gross)^2)
  \Delta Compton = \frac{\sqrt((Gross_a + Gross_b)))}{2}\mul (b-a) = sqrt((G_a+G_b)(b-a)^2/4)
  \Delta Gross = sqrt(\sum_{i=a}^b Gross_b)
  """
  s = spe["counts"]
  gross_sum = np.sum(s[a:b+1])
  sa = (s[a]+s[a-1]+s[a+1])/3
  sb = (s[b]+s[b-1]+s[b+1])/3
  area = gross_sum - (sa+sb) * (b-a) / 2
  # we've normalised the data; unc is only valid for gross data hence the \div sqrt(t)
  unc = math.sqrt(gross_sum + (sa+sb) * (b-a)**2 / 4 + math.sqrt(np.sum(background['counts'][a:b+1]))/background['alive_time']*math.sqrt(spe['alive_time'])) / math.sqrt(spe['alive_time'])
  return (area, unc)

def smooth(spe):
  """Smooth with Gaussian filter."""
  return { **spe, "counts": gaussian_filter(spe["counts"], sigma=3) }

def load_spes():
  """
  Creates a filename-indexed dictionary of normalised spectrum dictionaries.
  Also adds filename :: String, absorber :: (String, Int), nuclide :: String fields.
  """
  normalise = subtract_background_and_normalise()
  filenames = [
    "152Eu",
    "152EuAl1",
    "152EuAl2",
    "152EuAl3",
    "152EuCu1",
    "152EuCu2",
    "152EuCu3",
    "60Co",
    "60CoPl1",
    "60CoPl2",
    "60CoPl3",
  ]
  def get_nuclide(filename): # "152EuAl1" -> "152Eu"
    return "60Co" if filename[:4] == "60Co" else "152Eu"
  def get_absorber(filename): # "152EuAl1" -> ("Al", 1)
    absorber = filename[len(get_nuclide(filename)):]
    if (len(absorber) == 0): return None
    return (absorber[0:len(absorber)-1], int(absorber[-1]))
  return { filename: {
    **normalise(parse_spe(filename + ".spe")),
    "filename": filename,
    "absorber": get_absorber(filename),
    "nuclide": get_nuclide(filename),
  } for filename in filenames }

def calculate_peaks(spes):
  """
  Calculates the peak channels and areas (with unc) (cnt/s), for every spectrum
  """
  # Min & max channels for each sample and energy level, by graph inspection
  channel_bounds = {
    "60Co": {
      "1.17MeV": (1530, 1605),
      "1.33MeV": (1730, 1810),
    },
    "152Eu": {
      # "39.9100 +/- 0.0500 keV": (25, 75),
      "121.7830+/-0.0020 keV": (150, 180),
      "244.6920+/-0.0020 keV": (315, 340),
      "344.2760+/-0.0040 keV": (440, 490),
      # "411.1150+/-0.0050 keV": (530, 570),
      "443.9760+/-0.0050 keV": (580, 620),
      "778.9030+/-0.0060 keV": (1020, 1070),
      # "867.3880+/-0.0080 keV": (1140, 1180),
      "1112.1160+/-0.0170 keV": (1270, 1320),
      "1408.0110+/-0.0140 keV": (1850, 1920),
    },
  }

  # Add peak field with (channel, area) tuples per energy level
  # eg for 60Co samples it adds peak_channel: { "1.17MeV": (channel, area), "1.33MeV": ... }
  def compute(spe):
    new_spe = { **spe, "peak": {} }
    for energy_level in channel_bounds[spe["nuclide"]]:
      new_spe["peak"][energy_level] = (
        peak_channel(spe, *channel_bounds[spe["nuclide"]][energy_level]),
        peak_area   (spe, *channel_bounds[spe["nuclide"]][energy_level])
      )
    return new_spe

  return { filename: compute(spe) for [filename, spe] in spes.items() }

def separate_series(spes):
  """
  Converts the dictionary of all spectra into a list of dictionaries containing lists of spectra
  Each dictionary in the list represents a different series (combination of nuclide & absorber)
  """
  # the thicknesses are cm, unc is negligible; density is g/cm^3
  series = [("60Co", "Pl", 0.95, None), ("152Eu", "Al", 0.608, 2.7), ("152Eu", "Cu", 0.6275, 8.96)]
  for i, (nuclide, absorber, thickness, density) in enumerate(series):
    spe_list = [spes[nuclide]]
    while True:
      filename = f"{nuclide}{absorber}{len(spe_list)}"
      if filename in spes:
        spe_list.append(spes[filename])
      else: break
    series[i] = {
      "nuclide": nuclide,
      "absorber": absorber,
      "spes": spe_list,
      "energy_levels": spe_list[-1]["peak"].keys(),
      "thickness": thickness,
      "density": density,
    }
  return series

def rounded_string(num):
  return str(round(float(num), 4))

def compute_linear_attenuation_coefficient (IX, X, IX_delta):
  """
  Computes the linear attenuation coefficient μ given a list of intensities (eg in counts/sec)
  and distances (where the first is assumed 0) and intensity uncertainties.

  Note: the below explanation is out of date. It does not account for the *2/(delta Ix^2) factor
  or discuss uncertainty calculation.

  The intensity x cm in with attenuation coefficient μ is:
    Ix = I0 * math.exp(−μ*x)
  We have a two lists of corresponding (Ix, x) pairs, where the first is (I0, 0)
  We want to find the value μ which minimises the quadratic cost
    C(IX, X) = 1/2 \sum_{(Ix, x) \in IX * X} \frac{(\hat{Ix} - Ix)^2}{(\Delta Ix)^2}
  where \hat{Ix} = I0 * e^{-μx}. Then
    dC/dμ = - \sum (I0 e^{-μx} - Ix) * I0 x e^{-μx}
  If we iteratively subtract \epsilon * dC/dμ (IX, X, μ) from μ, we approach the correct value
  This will reach the global minima as the function is monotonic
  """
  IX = np.array(IX)
  X = np.array(X)
  IX_delta = np.array(IX_delta)
  I0 = IX[0]
  epsilon = 0.0001
  μ = 0.5
  iterations = 20000 # the leading digit is the max drift from initial value
  df = IX.shape[0] - 1 # degrees of freedom: data points minus no fit params
  def ReducedKaiSq(μ):
    IX_hat = I0 * np.exp(-μ * X)
    return np.sum((IX_hat - IX)**2 / IX_delta**2) / df
  def dKaiSqdμ():
    IX_hat = I0 * np.exp(-μ * X)
    return -2 * np.sum((IX_hat - IX) * X * IX_hat / IX_delta**2)
  for i in range(iterations):
    μ_prime = dKaiSqdμ()
    μ -= epsilon * (μ_prime / abs(μ_prime))
  # now mu is correct, we need to compute uncertainties
  # The uncertainty of mu is the span x2-x1 such that KaiSq(x1)/df = KaiSq(mu)/df + 1 = KaiSq(x2)/df
  # and x1 < mu < x2
  def walk (eps, mu, target):
    for i in range(iterations):
      mu += eps
      if ReducedKaiSq(mu) >= target: break
    return mu
  x1 = walk(-epsilon, μ, ReducedKaiSq(μ) + 1)
  x2 = walk( epsilon, μ, ReducedKaiSq(μ) + 1)
  return (μ, x2 - x1)

def calculate_attenuation_coeffs(spes):
  series = separate_series(spes)
  for data in series:
    nuclide = data["nuclide"]
    absorber = data["absorber"]
    # data["spes"], data["energy_levels"]
    layers = len(data['spes'])
    data['coeffs'] = {}
    for energy_level in data['energy_levels']:
      # areas : [(area, uncertainty)]
      areas = [data['spes'][i]['peak'][energy_level][1] for i in range(layers)]
      thickness_per_layer = data['thickness']
      thicknesses = np.arange(layers) * thickness_per_layer
      (mu, mu_unc) = compute_linear_attenuation_coefficient([x[0] for x in areas], thicknesses, [x[1] for x in areas])
      data['coeffs'][energy_level] = (mu, mu_unc)
  return series

def main():
  spes = calculate_peaks(load_spes())
  series = calculate_attenuation_coeffs(spes)

  # # Graph specific normalised spe files
  # plot_channel(spes["152EuCu3"])
  # plot_titles(
  #   "$^{152} Eu$ $\gamma$-ray spectrum",
  #   "Detector channel number (1-4096)", "Counts per second"
  # ) ; plt.show()

  # # Display calculated peak positions
  # last_nuclide = None
  # for filename in spes:
  #   if (nuclide := spes[filename]["nuclide"]) != last_nuclide:
  #     print("\t" + "\t".join(spes[filename]["peak"].keys()))
  #     last_nuclide = nuclide
  #   print(filename + "\t" + "\t".join([rounded_string(x[1][0]) for x in spes[filename]["peak"].values()]))
  #   print("unc" + "\t" + "\t".join([rounded_string(x[1][1]) for x in spes[filename]["peak"].values()]))
  #   print("channel" + "\t" + "\t".join([rounded_string(x[0]) for x in spes[filename]["peak"].values()]))
  #   # for energy_level in spes[filename]["peak"]:
  #   #   (channel, area) = spes[filename]["peak"][energy_level]
  #   #   print(f"  {energy_level}: {area[0]} +/- {area[1]} cnt/s at channel {channel}")

  # # Graph calculated peak positions with each series
  # series = separate_series(spes)
  # for data in series:
  #   nuclide = data["nuclide"]
  #   absorber = data["absorber"]
  #   plot_titles(
  #     f"{nuclide} $\gamma$-ray peak areas with {absorber} absorber",
  #     "No. layers of absorbers", "Counts per second"
  #   )
  #   plot_peaks(data["spes"], data["energy_levels"])
  #   # for spe in data["spes"]:
  #   #   plot_channel(spe)
  #   # plt.show()

  # # Print attenuation coefficients
  # for data in series:
  #   energy_levels = data['energy_levels']
  #   coeffs = [data['coeffs'][x] for x in energy_levels]
  #   print(data['nuclide'] + ' ' + data['absorber'])
  #   print('\t\t'.join([str(x) for x in energy_levels]))
  #   print('\t\t'.join([rounded_string(mu) + "+/-" + rounded_string(unc) for (mu, unc) in coeffs]))

  # Plot attenuation coefficients against energy level
  for data in series:
    if data['nuclide'] != '152Eu': continue
    energy_levels = data['energy_levels']
    coeffs = [data['coeffs'][x] for x in energy_levels]
    energy_levels = [(float(x.split('+/-')[0]), float(x.split('+/-')[1][:-4])) for x in energy_levels]
    # now energy_levels and coeffs have type [(value, unc)] so we can plot them
    mapFst = lambda xs: [x[0] for x in xs]
    mapSnd = lambda xs: [x[1] for x in xs]
    plt.plot(mapFst(energy_levels), mapFst(coeffs), 'bo')
    plt.errorbar(
      mapFst(energy_levels), mapFst(coeffs),
      xerr=mapSnd(energy_levels),
      yerr=mapSnd(coeffs),
      linestyle="None"
    )
    # plot the literature values for reference
    # shape is (n, 2); type is [[keV, cm^-1]]
    literature = np.genfromtxt(fname=f"{data['absorber']}XCOM.tsv", delimiter="\t")
    plt.plot(literature[:,0], literature[:,1], 'yo')
    if data['absorber'] == 'Cu':
      plt.ylim(0, 3)
    elif data['absorber'] == 'Al':
      plt.ylim(0, 1)
    plot_titles(
      f"Linear attenuation vs photon energy level for {data['absorber']}",
      "Energy level ($keV$)", "Linear attenuation coefficient ($cm^{-1}$)"
    ) ; plt.show()

  # # Save all graphs
  # for filename in spes:
  #   fig = plt.figure()
  #   plot_channel(spes[filename])
  #   nuclide = "$^{152} Eu$" if spes[filename]['nuclide'] == "152Eu" else "$^{60} Co$"
  #   absorbers = spes[filename]['absorber']
  #   if absorbers:
  #     s = "s" if absorbers[1]>1 else ""
  #     absorber_name = {"Pl": "plastic", "Cu": "copper", "Al": "aluminium"}[absorbers[0]]
  #     absorbers = str(absorbers[1]) + f" layer{s} of " + absorber_name
  #   else: absorbers = "no absorbers"
  #   plot_titles(
  #     nuclide + " $\gamma$-ray spectrum with " + absorbers,
  #     "Detector channel number", "Counts per second"
  #   ) ; plt.xlim(0, 2000)
  #   plt.savefig('./spectra/'+filename+'_spectra.png')
  #   plt.close(fig)
  # plt.show()
  #
  # # Graph background
  # plot_channel(normalise_counts(parse_spe("Background.spe")))
  # plot_titles(
  #   "Background radiation",
  #   "Detector channel number (1-4096)", "Counts per second"
  # ) ; plt.show()

main()
