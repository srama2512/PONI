We use Gibson and Matterport3D datasets for our experiments. Please find below the instructions to download them.

## Gibson
URL: http://gibsonenv.stanford.edu/database/ </br>
Target dir: `$PONI_ROOT/data/scene_datasets/gibson_semantic` 

* Download the Gibson scenes by agreeing to the [terms of use](https://github.com/StanfordVL/GibsonEnv#database).
* Semantic annotations for Gibson is available from the [3DSceneGraph](https://3dscenegraph.stanford.edu/) dataset.
* Convert the semantic annotations to habitat format following instructions in [habitat-sim](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#gibson-and-3dscenegraph-datasets).


## Matterport3D dataset
URL: https://niessner.github.io/Matterport/ </br>
Target dir: `$PONI_ROOT/data/scene_datasets/mp3d`

* Download the MP3D scenes by agreeing to the [terms of use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) and using the offical [Matterport3D](https://niessner.github.io/Matterport/) download script. Note that this gives basis-compressed versions of the MP3D scenes, which supports fast simulation with efficient memory use.
    ```
    python download_mp.py --task_data habitat -o $PONI_ROOT/data/scene_datasets/mp3d
    ```
* For extracting semantic maps, we require the non-basis compressed versions of the MP3D scenes. This can be obtained by downloading the raw MP3D scenes and converting them to GLB.
    ```
    python download_mp.py --type matterport_mesh -o $PONI_ROOT/data/scene_datasets/mp3d_uncompressed
    ```
* Conversion to GLB can be performed using trimesh. Note: The *.ply files from `$PONI_ROOT/data/scene_datasets/mp3d` can be copied over to `$PONI_ROOT/data/scene_datasets/mp3d_compressed`.
    ```
    import trimesh
    scene = trimesh.load(obj_file_path)
    scene.export(glb_file_path)
    ```