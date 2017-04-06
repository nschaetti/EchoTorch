

import echotorch.nn


if __name__ == "__main__":

    # Create Echo State Network
    esn = echotorch.nn.Reservoir(input_features=40, reservoir_features=100, output_features=10, bias=True)

# end if