import os
import easyvvuq as uq
import chaospy as cp

# author: Wouter Edeling
__license__ = "LGPL"

# home directory of user
home = os.path.expanduser('~')
HOME = os.path.abspath(os.path.dirname(__file__))

# Set up a fresh campaign called "grid"
my_campaign = uq.Campaign(name='grid', work_dir='/tmp')

# Define parameter space
params = {
    "k_min1": {
        "type": "float",
        "min": 0,
        "max": 21,
        "default": 0},
    "k_max1": {
        "type": "float",
        "min": 0,
        "max": 21,
        "default": 21},
    "k_min2": {
        "type": "float",
        "min": 0,
        "max": 21,
        "default": 0},
    "k_max2": {
        "type": "float",
        "min": 0,
        "max": 21,
        "default": 21},
    "out_file": {
        "type": "string",
        "default": "output.csv"}}

output_filename = params["out_file"]["default"]
output_columns = ["E"]

# Create an encoder, decoder and collation element
encoder = uq.encoders.GenericEncoder(
    template_fname=HOME + '/inputs/ocean.template',
    delimiter='$',
    target_filename='ocean_in.json')
decoder = uq.decoders.SimpleCSV(target_filename=output_filename,
                                output_columns=output_columns,
                                header=0)
collater = uq.collate.AggregateSamples(average=False)

# Add the SC app (automatically set as current app)
my_campaign.add_app(name="grid",
                    params=params,
                    encoder=encoder,
                    decoder=decoder,
                    collater=collater)

# Create the sampler
vary = {
    "k_min1": cp.Uniform(0.0, 21.0),
    "k_max1": cp.Uniform(0, 21.0)
}

my_sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=2,
                                   quadrature_rule="G", sparse=False, growth=False)

# Associate the sampler with the campaign
my_campaign.set_sampler(my_sampler)

# Will draw all (of the finite set of samples)
my_campaign.draw_samples()
my_campaign.populate_runs_dir()