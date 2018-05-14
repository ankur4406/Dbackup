#if __name__ == "__main__":
#    import sys
#    sys.path.append("../")

from form_slicing_v4 import form_slicing, clean_flow

def slice_coverter(marker_json, folder_path, flow_file_id):
    rec = mdb.find_one(DEFAULT_DB_NAME, DEFAULT_COLLECTION_NAME, {'flow_file_id': flow_file_id})

    # marker_json = {'Demographic_Change_Only': {'confidence': 0.9599798321723938, 'slice_file': 'Demographic_Change_Only_slice.png', 'slice_xy': {'y2': 3243, 'x2': 2031, 'x1': 1561, 'y1': 3200}, 'type': 'omr', 'lbl_align': 'right'}, 'Tax_Information_Tax_ID': {'confidence': 0.9751461148262024, 'slice_file': 'Tax_Information_Tax_ID_slice.png', 'slice_xy': {'y2': 6089, 'x2': 643, 'x1': 193, 'y1': 6058}, 'type': 'txt'}, 'Tax_Information_Tax_Name': {'confidence': 0.9751461148262024, 'slice_file': 'Tax_Information_Tax_Name_slice.png', 'slice_xy': {'y2': 6133, 'x2': 1462, 'x1': 843, 'y1': 6052}, 'type': 'txt'}, 'Tax_ID': {'confidence': 0.9571549892425537, 'slice_file': 'Tax_ID_slice.png', 'slice_xy': {'y2': 3156, 'x2': 1918, 'x1': 1468, 'y1': 3131}, 'type': 'txt'}, 'Tax_Information_NPI_No': {'confidence': 0.9751461148262024, 'slice_file': 'Tax_Information_NPI_No_slice.png', 'slice_xy': {'y2': 6089, 'x2': 1937, 'x1': 1637, 'y1': 6058}, 'type': 'txt'}, 'Tracking_No': {'confidence': 0.9661063551902771, 'slice_file': 'Tracking_No_slice.png', 'slice_xy': {'y2': 3025, 'x2': 1962, 'x1': 1556, 'y1': 2981}, 'type': 'txt'}}

    exisiting_coords = rec['coordinates']

    exisiting_coords_copy = deepcopy(exisiting_coords)

    for key in exisiting_coords_copy:
        if key == 'Form_Header':
            del exisiting_coords[key]
            continue
        slice_xy = marker_json[key]['slice_xy']
        coords = {"coordinates": [
            [slice_xy['x1'], slice_xy['y1']],
            [slice_xy['x2'], slice_xy['y2']]
        ]
        }
        if "coordinates" not in exisiting_coords[key]:
            exisiting_coords[key] = coords
        else:
            exisiting_coords[key] = coords
    mdb.upsert(DEFAULT_DB_NAME, DEFAULT_COLLECTION_NAME, {'flow_file_id': flow_file_id},
               {'coordinates': exisiting_coords})

    slices = {}
    for key, marker_configuration in marker_json.items():
        if key == 'Form_Header': continue

        field_type = marker_configuration['type']
        field = {'cut_path': folder_path + marker_configuration['slice_file'],
                 'field_name': key,
                 'field_type': field_type}

        if field_type not in slices:
            slices[field_type] = [field]
        else:
            slices[field_type].append(field)
    return slices


def slicer(flow_info):
    flow_file_id = flow_info['flow_file_id']
    nstatus = mdb.find_one(DEFAULT_DB_NAME, 'form', {'flow_file_id': flow_file_id})['status']
    try:
        if (nstatus == 'error' or nstatus == 'processed'):
            return False
    except:
        return True
    if not validated_flow_info(flow_info): return None
    history = None
    if "processors_name" not in flow_info:
        flow_info["processors_name"] = []

    try:
        # raise Exception("Intentional")
        for reverse_index in [i for i in range(len(flow_info["resources"]) - 1, -1, -1)]:
            history = copy.deepcopy(flow_info["resources"][reverse_index])
            # raise Exception("Intentional after history")
            if "slices" in history:
                return
            if "classification" not in history and "preprocessed_img_path" not in history:
                continue
            preprocessed_img_path = history["preprocessed_img_path"]
            classification = history["classification"]

            config_type = classification["config_type"]
            form_type = classification["form_type"]
            # classification["configuration"]
            # preprocessed_coordinates =
            # import pdb;pdb.set_trace()
            slices = {}

            if config_type == "border":

                #configuration = history["preprocessed_coordinates"]
                configuration = mdb.find_one(DEFAULT_DB_NAME, DEFAULT_COLLECTION_NAME,{"flow_file_id": flow_info["flow_file_id"]})["preprocessed_coordinates"]

                for field_name in configuration:
                    # field_name = "Patient_name"
                    # field_name = "1_Nature_of_illness"

                    source_folder = history["source_folder"]
                    field_configuration = configuration[field_name]
                    field_slice = extract_fields(preprocessed_img_path, field_configuration, field_name,
                                                 flow_info["flow_file_id"], source_folder)

                    field_type = field_slice["field_type"]
                    if field_type in slices:
                        slices[field_type].append(field_slice)
                    else:
                        slices[field_type] = [field_slice]

            elif config_type == "marker":
                ## Call Ankur code here.
                source_folder = history["source_folder"]
                # rec = mdb.find_one(DEFAULT_DB_NAME, 'forms_config', {"form_type": "CSF", "config_type": "marker", })
                rec = mdb.find_one(DEFAULT_DB_NAME, 'forms_config',
                                   {"form_type": form_type, "config_type": config_type, })
                field_configuration = rec["configuration"]

                folder_name = source_folder + flow_info["flow_file_id"] + "/" + "cuts/"
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                op = form_slicing(preprocessed_img_path, field_configuration, folder_name)

                slices = slice_coverter(op, folder_name, flow_info["flow_file_id"])

            elif config_type == "GCS8_flow":
                configuration = mdb.find_one(DEFAULT_DB_NAME, DEFAULT_COLLECTION_NAME, {"flow_file_id": flow_info["flow_file_id"]})[
                    "preprocessed_coordinates"]

                for field_name in configuration:
                    # field_name = "Patient_name"
                    # field_name = "1_Nature_of_illness"

                    source_folder = history["source_folder"]
                    field_configuration = configuration[field_name]
                    field_slice = extract_fields(preprocessed_img_path, field_configuration, field_name,
                                                 flow_info["flow_file_id"], source_folder)

                    field_type = field_slice["field_type"]
                    if field_type in slices:
                        slices[field_type].append(field_slice)
                    else:
                        slices[field_type] = [field_slice]

            history["slices"] = slices
            return handle_success(flow_info, history, "SLCR")

    except Exception as e:
        return handle_error(flow_info, history, "SLCR", description=str(e))

