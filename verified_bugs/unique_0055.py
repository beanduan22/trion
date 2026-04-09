#!/usr/bin/env python3
"""
Bug #0055: jax.jit produces wrong output vs pytorch_eager.

Patterns : [['constant', 'mul_zero_elim'], ['fusion', 'relu_add_relu'], ['broadcast', 'elu_scale'], ['branch', 'where_mask_fill'], ['normalization', 'variance_whitening'], ['branch', 'glu']]
Divergence: rel L2 ≈ ~0.00e+00  (jax.jit vs pytorch_eager)

Dependencies: numpy onnx jax torch onnx2torch
Run: python unique_0055.py
"""
import os, sys
import numpy as np
import onnx
from onnx import numpy_helper as _nh
import torch, onnx2torch

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unique_0055.onnx")
INPUT = np.frombuffer(bytes.fromhex("15ae1d407ad1e63f5ed19f3ff2180d3f6ac7323f551480bf20aa1abeee9c93be1af477bf26480c3fce64fb3e28949ebf950b7cbf7f0a54beded8bb3fdd82e63fddd3e43d4c958f3e1453e73ea23633bfbde350bf842901405289b23f4ad504c05d9e11bfaaf8bebe4ab492bf182e3c3e1aae0fbf31c42cbf399211c03b5e49bd8301a2bf91c27cbf3c68dcbe6a8e91be3aaff83f913f4a3fb8ea1f406673543e80f0483e32ad5fbe0a46ae3f770df5bf3fa521bf961493be3624094084ea34bfa8e0bd3fc5fb1dbf896b483f22bed7bee849a5bed2325d3eb43440bf126ee53ee5f276bfbdf0a4bf201731be319e463e96030abf91618bbfd0143e3f058c09bece755ebf57bed93ffca8613e94f357bfb2e6d53f0d14033fdb5281bf526b293f859bdbbe18e49ebf1c8bac3fa571babec50cb0bf255c593e1c8ab6bfbba602bffef14bbf0c04da3ed1a0933fdedd3a3e51bc4dbee6c34b3e5dee643d8794a9bfe01122becfb2e33ffe0899be6cbbc43fa17a98bf6a1583bfed14093eb28455bfb4824bbf1d64853f37ad163cdbac43be6c00853d92abc3bf9b2e1b3e5cb7a53f6d3f7ebce57af6bf9251e23e0b877e3fa9258d3f7ff9343fe5a6423f3deec6bebd5ccabee89ee83ecb8c10bf862f75bf7cf5823f96a4b23da3cc98bff98c7b3f35b3acbfd92ab03e5908833f010b7b3fb7602f3fd3cd0c3f1dd1053bf73ca3bf97da86bee64cd4bf20c0ea3efe3c4c3fe649413f86edebbf1b65babf7642813ef29dc03f25759d3f97ef18bf7353de3f4a5b843fcc31f13d11e6b5bf8065a8bf43316bbff80c10bfe99b54bffcad5a3f17bd5f3fba982fbfab6936be9f62b13fc7a80340570a9e3d44148a3f3491823f2bfab5be8933903e4ad58b3e22d4b03f32ec2940d083063f6c0457be5cacdcbf7498fabdf99cec3db9a3cebf457b12bf07da8dbee8f9383e75b5efbe0968813f02068ebf0fb3583f2c433f3f07b7743dddfad8bea7cce1bf1bb70040ea4d7b3e3ce70cbfe55e05c0ba2717be191e37bef16b0940008b0d40b5a6f53fe70e20bf3c1da0bf9046bebd6a8a6cbdb5d784bf81f0ba3e58a0e6bfd9b2b53f804a143fc65e5dbc68e586bfd0f3003f64798b3f7d2f06405b656e3e7e93523ff2d23d3f382a14bf9d81333f2dbc7b3f307a823f3552ea3eb39591bf9951f03de753013feef7933f67aed0be5385763ddf5b3dbe978815bf13554e3f60845f3d656d433ea73d573e69832bbf867874bd4cd28f3f1f7ea23f2163ad3f5fe2d3bf3f7ba7be8ee5993f1cc65c3f324cfc3ee9a3c8bfd7c8643fe08adbbf8a77bbbfb323fdbe0bca2fbee97bef3e9dd7133f0663e53f94faf5be6a3b79bfcba30640ffbe37bd25eea2bfa824bc3fabf73c3fb48c103e821bb2befe9b583ff5fd873f5cbaf23fe795e83f49ff1e3fac874f3fdf77883f9c11d9bfa6da763dab9cd3befa1348bf3009d6bfff9bc7be6df2b7bf294702bfd01b1e3fe98e0cbf2417303e438c883ff40e48bf79f8cdbeaa3e343f84209d3d18865ebeda6ea73e3b06883f36f27cbfccf10bbe153d083f5575e03e9015393f3c25abbf843fae3ff426dfbe0c0f7c3f18cf9dbffe289c3f95a1c63e0d74d7bff9bd2abfb903ccbfed9ff83e525b3c3f161b2a3fbc17263f98db89befb5bf7bfa955533fb5da0e40ce3f1abed68573bf5f8ddf3f70c3bb3ef777cb3eeaa3ad3d791b41bf3ec33cbf1173673fd13783bf198572be573c4b3f35fad53e917250bc010762becc1b153e632edebf258cc4bfdbe38f3f6ccaafbf4fbe0d40e94465bdb2be423fe500b73f2bd394bf2403023f1eaac83fcfd569bfe6c691beedbffdbf4d775d3fbed15ebebdc9953e98ccafbfae2d35bfdea6bfbffde442bff3661cbf256e1b3fde55c5bec14a9b3f9bae493f9812e6be77585c3f37f418bff459b13fcea7b33ea482fd3fef297fbe6e3c1cbeabb0a9bf5b4187be782cf4bf3c5ba53f035ae43c37fba23edb1c403e7917d6bf2c1b893eb820ad3fe59fc1be211a3bbd9129613f673d95bcd6195bbe179d63bfb430b3bd1ca0d3bdfab4d1bf071e633f715633bf00d3103da71ab33f0478d53ed146d5bff300663ff5113cbfcb1353bfd92f853f30625fbf9db7a53e7632e4bebe5b6f3f249df63eb141773faf9f013f20732dc023f7203fecc71e3d717924bfe006d63f6f46a93f1dabbb3fe5fcafbf9ce4d03f053468bf6515ac3da64a003dcaa424bf598eb5bfaeef72bf877b15beffad963fe904673f7ea9acbf3b75b4be11f2b7bc458500bf48f10bbbc5618fbfdf9caebf98a835bfb60b45bf041e6bbfaa7997bd085a23bfb08f58bf6a4da93e2d36c3bfaf3bd33f4f222a40e01a6e4068051c3f3c47923d8c1d19bea5e909bf142d813e54d198be11318e3f570f2c3fabaaa4bef676b8bfc0689c3f6151f13f92e36f3e238becbe3ba51b3f22f238be3720c0be619d8e3ed3e3913e1202f6bef69695bf70c614bf75b8bd3f4c2d28bf48cf45bf3c39bc3ff38a33c00b2d70bfaeb426405aedafbec1579b3ff1dd283f5aa5aa3f1372e53e36808fbccf6d01bf979ed83ea77f623f33aa333f600dd1be27b07a3f4af9e5bf85a7f93de77e793f0d4e2cbe071817bea87422c0f418c8bf49daa83f1bea50bff8f9b2bf7f0d7a3fbb976c3fa9510a3f7a0c29bf46d7733f0b13e9bf0343e7bfc64482bfcda6febf6232593f674ac8bedf92d53fdc598e3f2cc4a43ee580bcbdf697e23e6611473c790b9a3f48ec82bf2ff83c3fab42b53f6a68ea3e2431e73faf11573fe43979bf5698273f7175a33e56c1dd3f5da3babf85466ebfaae5eb3d1ab552bf280293bf1150c93dd6ec73bea2bfa8be8ef6233f4539b5bf912f0a3ef823a03efce0d63f49bec9bdc7f09ebe118edfbf30a8563e82d04f3f518efbbcd1bdcabf569b113deb01d63ea9df713f9c3f01c0c6daa43f20368b3ef0a19d3f1b8b6ebf2d3bc0bcf46476bf734dc8be537abdbf769e623f22e78dbe9411d2be6bfe743f61c4ba3e3bb648bfcb318cbe5253f93ea7216e3ef4e0bf3f2203243f2938f5bf3873c73fe71f42bf969730c053afb1be577e02c0e5127cbf07490a3fda3b23408afd503f5f1fe43fd181e53f2eb1633fbfd499be1685c93fe82d633f0170c4beee11acbec078b93f87ffa63f7e5ab6bfc85ab5bd505a88bfa9002240d71a2740d1defebb61cca03f081630bf13c6a3bfaef4f2be01006cbfbfa4993f545da63d7f42f2bd3b4ee9bd28fa95be0f729bbf50b6003f670414bf2b01b4be1c2ed5be63d88c3f65a935bf50e2d1bd9d254fbf2ac5963d1d62acbf06b3423fa0902fbf97f3253f55653e3f89b4f2be2f63113f4e3c473f319129c078e7953ed5ca09bfa4ed35bf6780343f3d47d1be15e554bec0c0093facbe8a3f2befdebf890383bf20fe6c3f4071d73dd9ab18bfb165cb3dc89ab93f75d5983f5b10a8bf3f0bbc3f4363c3bf0593b43ef249e0beb28ec63e8db5df3f025541be3097403ffc12afbf2d9d9fbd63d505bf941579be01bac7bfc5281d3fa5040b3f7090f4bf6bac1dc0f5aa34c0066271be63f53f3f3b2275bebacdb53f0c3807bfbf06e23ea4c7fabf480cf03fee4322be0d8102ba42e8973fbd98a63e2d00ea3f7b4015bfb108483f8d2df73cbe56bdbf637167bd05c9943eabdfaebfb01c923fec78e2bfda4a143f450e31bfc670083fa06ddc3b2de2223ee9d90b3fc118a23fdf7be33f9e3ca63f2898b1bebd4871bf05ff8f3e8e56633ed75a0c3d5cd25fbf96e1013e1c6d9bbfa94ad03ef29495be37bbca3a0644d4bf94ae1f3d15f6bcbe2cf1dc3e6f8911beccda15c0e7354c3f0a1da6bf517997bf2f7a9e3e77eeb03fbd5fa43f7b6800bf631b2d3f565d00bf0ae3f6bf586652be5bc094bf997e02bf796dd03ede4b043fe4a84b3eb7fd8f3fee0d73bfd621b43ee2ccb83f14d38dbf3bac6c3ff38ef13f6baeb4bfdb25a2bce2ab8bbf8826bcbfb07b45be4c0de63f5af1b8bcd2feb73f9220f83e28303ebedaad98bd790806c01f039dbf17628f3f981c2fbfe8ed9cbe2180aa3ef92b1ac08d0059bf17fbafbfddd55fbdd3e384bf648d2fbf67d53b3de4cf48bfa680743fde900e3faa5cc93f9fb69abf8e6f91bfd8b24abe1706333f9ee6893fc6800740c56c20c0d13fa93f33f68dbe22f4c9bf3f7019bfeffa823e92a8ad3f80a9a73f5905f2befcfeda3ede01c53eb90dc7bf85aebbbcb804a1bfe987ae3f451eb33f6044723ff181d63d1c52ddbec6010abf9e5debbe696f8dbf3b60c83f9b49d8bfa7c3b03dbd3b8c3f32d30abfecc0753f18d1b23f82b9c5be869892beb1ad01be4e59d53e2ccaf8bf4706a23e8e489cbff5bef9bfc1ba39bf495b49bed19f8a3f71c421c0414b803f7be24ebf6468743c7ab0643fdef76abf2666064085bb053f3f2c53bf5267813ef300f83d3c2c6ebf420850bf1c75a6beb9be0abff15414bd26e3883ea551763f61370abffbc3023f987fac3fabde883e3bbdab3fde2c373f34ece3bf91cb8f3e92bc17bf20ab47bf107c9f3f0035933fa73f6cbe80818d3e6ab588be59565dbf95ebba3f8358483ff054093eed18a03fa29f71bf13dc1d3efbdd1b40702212bfa1db603fafd3b4be3245b53ffb1a743f327e123e8a6966bfb374e13eea2a23be4117853e4f2f5fbf3dea0e3f908d9d3f11d1073fdea5f3bd8bbafebf00252e3fd95fecbe7e71603e9eb038be940934be2f19d0bed389043e5ecf143ff010643f3ccbe23eb60fef3f346c983e4ecc863e11fab9bfe49bf13fff72503e13aeca3eee666fbd16faecbebae0543f71c437bf60d397bef00a41bf3f3f22be29c2983e0f733c3fefdf57bdf9a355bfdfcbb5bf91bcf43e5c62c4bd5b26b1bf48254f3f8bc45ebec8af17403ca4d53f50f5d5be90686cbf88ddd4be1d219e3fb267943e72f8003f93ef1b3f927b493fec5dd53ff07f10406117293fdd712dbf6075b5bf64e4cabe736345be99f803c0c03fb13e610534bf1083a2bf00bc90bec86d4e3e3a18af3e6b8da4ba75367d3d81ac26bf916b2cbf48700a3e211bf4bf2f023a3e859709bed362963fb85a213fd025643f0a23ddbf159a203fb4df5d3e0eb8b73da4b7b5bfe0da12bf1b26b73fb33d23be071aa93f375f8e3f2235bfbebfb8fd3ef5210fbfc2636dbfb8dfc8be19c5f9be92f3a6be548aeabe90fd383f5c22d93f8b481540593ba4bfae4c05bf2af5653f11a2133f45bf4e3e52d8f23fc463a0be403528bd0d464f3f092abb3ff96473bf23a04f3fd2315a3e3a08e0be8fdfa13d9fd7cd3dedb0933ebd315fbe772863bffb019ebf75721dbf3523ff3f4e2ea33e51a8d4bf02ff15bfc74c8ebe1b19623e91c8cebcc48d503f454d9b3e57b6e43ecbd4a93f370bfe3e33ead43f7f6d5b3fecb5d73fc733fabeb26b813f552829c00672d8be030a4b3fab32a63f0144983e1b7a853f3e92a1bfa0457b3e47901a40c7edfcbfaded37bed180ad3f31e57abebfb0203e332f543f67f086bcf93b6e3f1758014046cbcb3f2cfa0cbf781cadbed94a2bbfd5f5923f0db0acbfabee553f361a9abe3555d6bc9f0a0f3f55fc90bebd93a2bf1ee0b6be6702c4bf09b963bfe00ac9be1a38983f82a407bf580c113fafad98bf6c99a33f97f3723fb1ff2bbe617e203f01ff8f3ed51e14c071a69dbfe03a6abe54b94d3fd57fe4bf53d4c0bf91835dbd5d4982beafc2643e0da721bf0bf3b23ff6dbf8bc9cce953e30b8783e024599bf5df9fd3fa1d52a3fc507ab3f1a84c4bff9a12bbfdef953bf036f98bf3a735ebe3e500c3ffbc9e6bf939aa3bd1114833f34cda4bea65f603f911595be4581c1be3fb42ac0a536763e8910a53fe1cf833f48afde3d93bb8abf399fdbbd75f4ebbf68e1a1bd230485bf9b7fafbfe3c8783fd62de6bf2cf2debf0c6cc93f51d49f3ea2c8c0bfc60f76bf07f2ccbfd1bc21bfa39b52bf752f28bfbee7dd3e20bb1d3eb14730bf65cf98bf2065013fde9c36407b5e63bf65c02a3f6a7e84bf51d5d53e37870cbfbf86cf3eb38585bfa825bcbffd5958bfc5e6cd3f63f1fb3f5dbaed3fb4db573ea2c40cbed97d4bbfa2ee8fbf73525c3ff781e13fa28b1d4017ab5cbfd0fe97bf0ffd783d900b2c40b6a3453e1fb6d43d32f9cabd72342740d172af3f01b73ec07967b2bfe4ac29bf597e063f5b7904bfa254d73f151c583f2545d7bfd7779dbfad5fdcbf4403b53f9d028abe9e1f993e527e8e3e3a10dd3a6129863fde91c43ecafb6a3f5ce1e03e6a48043f3b0a8e3f70a11b40ea6e7e3ec8b14d3e6ec1ecbeb148f83e4268b0bfcf6424bf6f9ae13f747df2be2413193f6d6d94bd5e1d73bf921e21c0d48aa53fdfc82cbe9439d3bf9b40583f2ee1acbf808952be3326d4bfe8620b3fafdd3b3ffdbf8b3f6329b4be6e93a1bf42bd7c3f738cce3e00ec49bf75e38a3f963e0840d75c183ed4e7c2bee64303be813bbfbe7e87c73eb989543c30251d3f8bc298bee6210240b80e3a3e40b39bbeb6b73e3f545da8bfaeb912bffeaa223f3516323e83bb393ea523f2be3b7f76bfefb4abbdf7175b3f965ab3bee32024bfede0103ff79c3b3f8f1cc1bdda4945bfdac899bfe7e5d63e4802b1bfc6bcb73890501040ee5c573da3a0d33f08953dbf7c358dbf1ea727bf13ec35bf31b56b3e43ec50bf7b4620bebc27833e226a1d405497d7bf617b8e3f6529663e35df0ebf77cfeb3e74f10d3fae44e03eaf1b403eeefa3fbfe585b63e70e28bbf7ad08d3f92d0b73e2119cfbd9019bdbf6c950bbefcdb243fff1c35bf869450bd837d23bfaf02f7bf6376bbbeb6b0ba3d6b5fd6bf2b3a653f041a2bbe2822ad3f4e682abf7ffd8ebfb6e4763e804d16bf28778e3fd69389bf82530ebe7b079cbffbd931bf651e97be75639c3f9b8fb73f0e1d65bea5e3ab3f0c0a853fd882053e86e82abf756dffbedbe6213e999c78be2596c8bedca4e43e195374bff584b1bf28c4ccbf653bdc3f730e1f4087f553bf27fa133f1e4b033e98e14abe531c26c006874bbf745590bf3065db3f953cfc3df16764bfcf60b63f32383d3fa4a9203f355b4dbd6697093f82022c3f0929983cae6ca1bff8c69cbef3d89abfc847623f65f3fcbe7e4844bf3bd2ce3dbc92e3bfe2eb8fbea8ad0bbf7f90683f9480fa3e9bc0dcbfa24193bf00e44cbc9719494045035f3f75ffc63fbcd0833fcfbb853ef8bc14409525f4bf1f7f7cbfbb2d793ffc8a67bf04980a3e516eb0be2ffb35bed9690cbdbf8b5abfdd32f2bec5d834bfd6d41d40cbe258bf17d7043eb209c6bb812097bfafd1a3bd38c237be18d97fbf6ff19b3fb72364bf2387bc3f8113c0bfc34ebebfb485bcbe9bda4cbe25e00940062499bf755a3d3eaa4caf3f88e4973f58bbc0bf6b2ea6bf52c38fbe6cde2bbab950563d47ff0a40fb67fd3e143f7cbd909ec0bfe8bd224074c8d23dd35a6fbf0b27bcbe4efca5bf7b75803e2eed54bed7b0b2bff2da4c3f5fa356bf25f295be31d2b63e285510c0664ddbbe825d3abfddc611bf4da767beae684f3f38dbb5beaf408f3e39fa4cbeab0fc3bf9ea6a9beba2b0bbf4c2f833fa955b2be1a61c93f2b92f83e9703533f9844e4be8f90e7be332962be74798c3e801e853effa0e8bd82ae6b3f0dad633da36c853f3509153f54d2db3fa05ea3bf8ffcb7bf10ad8f3edd75e23f68e143bfd902593daf41253f76ad29bff9c6f1be856fc13e54209bbf8050ae3f0aed67bf3d7ab83ecc31d9bfc07688bf54953fbc8ed12d3f5e5d25bf3c22afbf2baf6f3fe9d07b3ff1a24f3f62584d3fe230a3bd63dcaa3fccbd4b3ef7c32ebfb8da813ebf43653ee6e8f83eac50e73fef6bcabfa377e33e10b785bfe3c71f3f3f0ba2bf5982993e117fbbbe2598fe3e7b7bf4bfd3df603fdda64c3ee0b59fbfdd4a273fde36c9bdf3d6ae3f032ef1be4d82383ef4af8c3f431b59bf145875bf96211fbed9bb47bec52c4cbfc28f1b3f00c0e23e97fe81bf5811713e923a8d3ff6b9373ea00a4d3eb735913eb4fbe63e1687423f9db570bf20f104be1aadcdbf7c8480bf8ad1eb3ea74122bf90e86b3eff6fa1be490314bfb7aba93f4717b0bf3abd5b3f69e220bf0a5a43bfc368bebe3388973fbe483a3f74eb95bfc3d44f3f400d233f19a6993fc821d73eecc9e53e42d946bfed6518bf6b2ad6bfd476da3d1214043f6874a93ef7f00fbf5d15fb3dccce0fc00bd63d3d8f1400bfa232efbf8ab71ebf3dc5803d931e49bf17ffafbf02b4b93ec5d118bd5c3d253d0a1d893f692e713f7e171940754311c0be63223f3f71f13f6d5cfbbfa303bb3e74bf01bff88aa53fd2f4d63e19c63c40a904433f87ee293f95c7f33e229910bfbfa6373fd9cad6becd7a93bd279c90bec05a99be1a4bee3e2daab1bf69e318bf5c0027bf4edfe13fb8a52fbf9d05663f9b2c9ebec97700c001c35abf9ec94e3e0e978fbf2592c73f00b43a3eb27f873f1bfc4ebd51a2d13fb03f9b3f6faca13e8c82d43e0f8286bf598731bf511befbd60dc49be4cdebc3ff45b70bfc149b1bf28a410bf6e88203e3af5b83f90373cbf6827763fa6d12ebf0e91ed3f817a40be38dd343c2897c63e56aa993f0182d7bf76c0303d2e9014bf8b657c3f2176e23f83804c3e014af4bf75878abf8577113e5442fabfeaab2d3e44a9633f34f2f2bf1bbf693f6d31e1bfae9cc03f72fdbd3f2633863fd87fa0bea4e42ebf2ed869bdee9716bf8df730be620100c04c1e973e60cf893e7320debff9aa3bbfa3e02a3f05d5bc3e29eef4be00e8743fc16711c01eadbebedb496f3e99fc8e3bb782c1bd7a7d963f06258abe8ad6183f2ae5b0be2922ce3f65409c3f171cd83ece4c223f571a8dbeee4b8c3f2f196fbe9f8e09be3d1085bf686e65bf90ddb7bf7c69c73e2b952d40f46830bf2c7d18be9814223fecf2a53cfba26bbf561bc1bf84b307bf5b030c3f06a75fbe6ae5d1be8e682e3f2bedb63ea9e0abbecd85cf3f631688bfce68a4bec36161be6c63f23f6d780b3f0da783bfee57913e0f5160bf7e9922be6e538abfc396b13fbd749f3f0278743fb76f4ebf4291cabe7f66c7bf76e1153eb9dbe53eac8280bfdc28a8bfb6b5853e0ec8fb3ea129993eb4688bbf68de8ebec7828a3f643bba3ea9be4e3e3ecd71bec226f2bec4c119bdf3c0523f9c308a3fd42c3c3f18e4963fe0a1c1bff2e2cc3ec92d413f2f7dab3e78ced93f2c5bf43ed304e7bf5fcd84bf37ab5d3f0e01003efcbe7fbfcecd19bff2d615406e53763f155e923fccbf0e40d00e3bbebff3abbe7e6218bf5830b23e60ba37bf4d49ab3f6fd28abf71df0cbf39a41bbf00f1083f81fecebe16b4963fb04398be649bddbe04c4323ff98c16bed378283ff322bcbe899b1abe230323bfdf72533f0d5de53fee6da8bf97499bbe8f2c28be56360d40eb9d473fe23310bf57f5633f7b6e263e405cfe3ee2c265bfa496cc3fea6d63bedfb9773f28524dbf1c978c3fc891923eebbeadbda1198f3ca8ebac3fbf391d3e2723733f0eb56dbff7df0c3f1918fc3f38d65d3fa79c07bea09febbe14723f3f61f6733e3d1c653f49f53dbd813d5b3fb630c2bee7eff23e589088bfa98215c0720fd13f273e04bf556a9a3f941fd1be61ca673f1bbf1bbf9fdc243e0cec9e3fa04d7bbdb6776d3ebd12873f2771863fe01e473fa64f003f85a10a40438ee43fd1b5213f6178183f3a5071bfec5d6bbf5bbaae3e6916d7be842e923f25b592bf77ba17bf416c8cbfb54616bfe004a03ebc6dca3dd3dc3c3e6d15193f54bdb5bf11d195bfc80075bf33d4bebf6a61a6bd48b00f3fa2e8b43fdbf7c43e2b6a2e3ff8ce4abfe3a9113f309aff3e07608abf0ff6813ebaf287bffe67f2bece865c3e717ebebff70b57be8d5033bf1fb6ed3e53ccd13ea060f4bcf163d9bde3056cbf41e3a43f24b79b3f529020bfe95b9ebdb9a2b83e717ffa3dd37d1b3f52e4e4bc3f2680bea7d4c7bff76882bfe25b643fb6eb4e3f998eb23f230e8cbf583045bf9b3fddbd587358bf748600bea034a1bef6501f3fb9d98abf84d306c0f9c31bbe1c0e83bf9229ec3d237b0bbe6058a63fd50807bf9ff6d43e0be94e3f85c14cbf5e9c9d3fb207e53ebcedc73f86f2a63f81fd2cbdb7fe53bf0d49603df72e4b3fb78accbf744f30bf4afddbbd0f99ad3e5f83ca3e5410583f3cddd73f3c169b3eaad907bf97b73dbf59e636bf4ff17bbfe4a7d23f05e7873f665ac3bedd1ddabf4851d43d0964ee3f8434a43f31e5e5bf6b242abf53d2bc3f8b4f40be2ffe34bfa2b6b9bffc2322c0c00598bee4f3a73f3466cb3fd5652abebdafcbbfa221883f4af343bec235b1bf478552bdaacdd73fe33081bfa3234dbf460b993fd5540cbf9864cf3e6d5c6bbfbe5415bf5f27523d08f4a83e28b0253fd6a4a9bf980aa6bf13e790bed8aee33e837f463f870fe1bf24fa15c06813813f8acfa6bebf7477bdb64d743e556dbf3f700df9bd08e180bfa6aca7bdaf75cabeee34d7bc255561bd715a8b3fa76ce03e6cf237bf5cee613db981d63f183d613f36a29ebf8fb8403e4ed41c3d1aadc1bf24718ebe2f2cb53f8d0b04bed90849bf82bf8b3f05e600bfed76b23eb00ae23e2be26e3fd0a205408923033ff5928e3e23eb353f7761fb3e80d106be946d76bd62f8bbbe2bbe753f0f6e57bdc78991bea47c6f3fba37c4bd273e68be7102f33f9738463f3f5ee63faffb12beda555cbfbde188bf9a91a63f3b34d43d52587abd48ccb3be7fc9a23f5d852d3e150c03bee453b3bf9323a3bf9e6036bedb3fca3d45c854c0a3cf713ff4511abf4ae314c023e7e9bcdc6ff1bf83e69a3aa901fabe1fce113f3b1c04c0b6c9edbffce02e3d30cd5abf8ea2d5be4657a73fd5ba48be182b69bf46a1483e8b1feb3e6ab3b2bf0f7d633e9ee9f53ecf4a773f7dc416402a2ab53f8e9c6a3f7d97f2be784b80bed74ccabfeb801fbfdb999abdf78dd0bd02431abf5f42ee3f3794ecbd54a6003f11aa633f623eb53f0544acbe44ec08bf02b9b83bb7d2763e0557543f70e0553fe76d07c0b4cf073f7f83b43e042df03d2272763ee187403fa594b7bf410eddbf4e2b83bf998bf73e9cb69f3ee46135be517decbecd95e93e03863ebf41dd91bf2a1d3cbf6fc410c06d3ca53f23a4d73eaf296f3fa8dd4fbfda930a3f969617401a0ac33ff81e963efd9e6abcd1f968bf71f1e23fd2f2fd3efd0fa33e9e5101bf0d06bcbf96a3afbf252d2f3df3fc85be6fb70040777d4a3f53f98a3f219d65bfaeb005bf8fba1a3ef1f179bee28096be28cbc1bf4305d53f65c2873f8d83853e094f2b3eae3690bd2cd801c049c68fbf13569abefe929cbd12d28abd9ff55d3f1d13ea3e5876273e94467fbef796b1bf3b5f8e3fd5c36e3fcf0891be4985fa3fd5de12bfe3c522bd8dfc283f1ecd863e77c3033e913905bf0605473f67a60b3f707f1cbf42b6863fd7ed3fbd4270113fe0231340270fa8bf2105c8be1a7268be4b1773be945447be75f2a03f8a379ebe068189bf0e9eaa3f46bbdcbb5cd7fdbed03bc13f795e38bff832a73ffd570d3e03ab26bf"), dtype=np.float32).reshape([1, 16, 128])


def reference():
    """pytorch eager — ground truth."""
    m = onnx2torch.convert(onnx.load(MODEL)).eval()
    with torch.no_grad():
        return m(torch.from_numpy(INPUT)).numpy().ravel()


def target():
    """jax.jit — under test."""
    import jax, jax.numpy as jnp
    model = onnx.load(MODEL)
    inp_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    inits = {i.name: _nh.to_array(i).copy() for i in model.graph.initializer}

    def fn(x):
        vals = dict(inits); vals[inp_name] = x
        for node in model.graph.node:
            for nm, v in zip(node.output, dispatch_op(node, vals, jnp)):
                if nm: vals[nm] = v
        return jnp.asarray(vals[out_name], dtype=jnp.float32)

    return np.array(jax.jit(fn)(jnp.array(INPUT)), dtype=np.float32).ravel()


if __name__ == "__main__":
    ref = reference()
    out = target()
    diff = float(np.linalg.norm(ref.astype(np.float64) - out.astype(np.float64))
                 / (np.linalg.norm(ref.astype(np.float64)) + 1e-8))
    print(f"expected (pytorch_eager): {ref[:6]}")
    print(f"actual   (jax.jit):       {out[:6]}")
    print(f"rel L2: {diff:.4e}")
    if diff > 0.001:
        print("BUG REPRODUCED")
        sys.exit(0)
    sys.exit(1)


# ── ONNX op dispatcher (required to run the model in JAX) ────────────────────
"""
Shared ONNX op dispatcher for JAX, TensorFlow, and other array-API backends.

dispatch_op(node, values, np_like) executes one ONNX node using the provided
numpy-compatible module (jax.numpy, tensorflow, numpy, etc.).

Rules:
  - Initializers are stored as plain numpy arrays in `values`.
    Shape-extracting ops (Reshape, Slice, etc.) call np.array() on them safely.
  - Intermediate computed tensors are framework arrays (JAX/TF traced).
    They are NEVER passed to np.array() — only used in framework ops.
  - The dispatcher is framework-agnostic: pass jnp for JAX, tf for TF, etc.
"""
import numpy as np
import onnx
from onnx import TensorProto

_ONNX_DTYPE = {
    TensorProto.FLOAT:  np.float32,
    TensorProto.DOUBLE: np.float64,
    TensorProto.INT32:  np.int32,
    TensorProto.INT64:  np.int64,
    TensorProto.BOOL:   np.bool_,
    TensorProto.UINT8:  np.uint8,
    TensorProto.INT8:   np.int8,
}


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.FLOAT:  return a.f
            if a.type == onnx.AttributeProto.INT:    return a.i
            if a.type == onnx.AttributeProto.STRING: return a.s
            if a.type == onnx.AttributeProto.FLOATS: return list(a.floats)
            if a.type == onnx.AttributeProto.INTS:   return list(a.ints)
            if a.type == onnx.AttributeProto.TENSOR:
                from onnx import numpy_helper
                return numpy_helper.to_array(a.t)
    return default


def _np(v):
    """Convert a value (numpy or framework tensor) to numpy. Only for initializers."""
    if isinstance(v, np.ndarray):
        return v
    return np.array(v)


def dispatch_op(node, values: dict, F) -> list:
    """
    Execute one ONNX node.
    F  = framework module (jax.numpy, tf, numpy, …)
    values = name → tensor (numpy for initializers, framework array for computed)
    Returns list of output tensors.
    """
    op = node.op_type

    def get(i):
        if i >= len(node.input) or not node.input[i]:
            return None
        return values.get(node.input[i])

    # ── Element-wise arithmetic ──────────────────────────────────────────────
    if op == "Add":       return [F.add(get(0), get(1)) if hasattr(F,'add') else get(0)+get(1)]
    if op == "Sub":       return [get(0) - get(1)]
    if op == "Mul":       return [get(0) * get(1)]
    if op == "Div":       return [get(0) / get(1)]
    if op == "Neg":       return [-get(0)]
    if op == "Abs":       return [F.abs(get(0))]
    if op == "Sqrt":      return [F.sqrt(get(0))]
    if op == "Exp":       return [F.exp(get(0))]
    if op == "Log":       return [F.log(get(0))]
    if op == "Tanh":      return [F.tanh(get(0))]
    if op == "Reciprocal": return [1.0 / get(0)]

    if op == "Pow":
        return [get(0) ** _np(get(1)).flat[0] if isinstance(get(1), np.ndarray)
                else get(0) ** get(1)]

    if op == "Erf":
        if _is_jax_module(F):
            import jax.scipy.special as jss
            return [jss.erf(get(0))]
        else:
            import tensorflow as tf
            return [tf.math.erf(get(0))]

    if op == "Sin":   return [F.sin(get(0))]
    if op == "Cos":   return [F.cos(get(0))]
    if op == "Floor": return [F.floor(get(0))]
    if op == "Ceil":  return [F.ceil(get(0))]
    if op == "Round": return [F.round(get(0))]
    if op == "Sign":  return [F.sign(get(0))]

    # Element-wise Max/Min (binary)
    if op == "Max":
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        result = tensors[0]
        for t in tensors[1:]:
            result = F.maximum(result, t)
        return [result]
    if op == "Min":
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        result = tensors[0]
        for t in tensors[1:]:
            result = F.minimum(result, t)
        return [result]

    # ── Activations ──────────────────────────────────────────────────────────
    if op == "Relu":
        return [F.maximum(get(0), F.zeros_like(get(0))) if hasattr(F, 'zeros_like')
                else F.maximum(get(0), 0.0)]

    if op == "LeakyRelu":
        alpha = _attr(node, "alpha", 0.01)
        x = get(0)
        zero = np.float32(0.0)
        return [F.where(x >= zero, x, np.float32(alpha) * x)]

    if op == "Elu":
        alpha = float(_attr(node, "alpha", 1.0))
        x = get(0)
        return [F.where(x >= np.float32(0.0), x, np.float32(alpha) * (F.exp(x) - np.float32(1.0)))]

    if op == "Selu":
        alpha = float(_attr(node, "alpha", 1.6732632423543772))
        gamma = float(_attr(node, "gamma", 1.0507009873554805))
        x = get(0)
        return [np.float32(gamma) * F.where(x >= np.float32(0.0), x,
                np.float32(alpha) * (F.exp(x) - np.float32(1.0)))]

    if op == "HardSigmoid":
        alpha = float(_attr(node, "alpha", 0.2))
        beta  = float(_attr(node, "beta",  0.5))
        x = get(0)
        return [F.clip(np.float32(alpha) * x + np.float32(beta), np.float32(0.0), np.float32(1.0))]

    if op == "HardSwish":
        x = get(0)
        return [x * F.clip(x / np.float32(6.0) + np.float32(0.5), np.float32(0.0), np.float32(1.0))]

    if op == "Mish":
        x = get(0)
        return [x * F.tanh(F.log(np.float32(1.0) + F.exp(x)))]

    if op == "Sigmoid":
        x = get(0)
        return [np.float32(1.0) / (np.float32(1.0) + F.exp(-x))]

    if op == "Softmax":
        axis = int(_attr(node, "axis", -1))
        x = get(0)
        x_max = F.max(x, axis=axis, keepdims=True)
        e = F.exp(x - x_max)
        return [e / F.sum(e, axis=axis, keepdims=True)]

    if op == "Softplus":
        return [F.log(np.float32(1.0) + F.exp(get(0)))]

    if op == "Clip":
        x = get(0)
        mn = get(1); mx = get(2)
        if mn is not None:
            x = F.maximum(x, F.asarray(mn, dtype=x.dtype) if hasattr(F,'asarray') else mn)
        if mx is not None:
            x = F.minimum(x, F.asarray(mx, dtype=x.dtype) if hasattr(F,'asarray') else mx)
        return [x]

    if op in ("Identity", "Dropout"):
        return [get(0)]

    if op == "Cast":
        to   = _attr(node, "to", TensorProto.FLOAT)
        dtype = _ONNX_DTYPE.get(int(to), np.float32)
        return [get(0).astype(dtype)]

    # ── Shape ops ────────────────────────────────────────────────────────────
    if op == "Transpose":
        perm = _attr(node, "perm", None)
        x = get(0)
        if perm is None:
            perm = list(range(len(x.shape)))[::-1]
        return [F.transpose(x, perm)]

    if op == "Reshape":
        x = get(0)
        shape_raw = _np(get(1)).tolist()          # always an initializer → numpy safe
        orig = x.shape
        shape = [int(orig[i]) if shape_raw[i] == 0 else int(shape_raw[i])
                 for i in range(len(shape_raw))]
        return [F.reshape(x, shape)]

    if op == "Flatten":
        axis = int(_attr(node, "axis", 1))
        x = get(0)
        left  = int(np.prod(x.shape[:axis]))
        right = int(np.prod(x.shape[axis:]))
        return [F.reshape(x, [left, right])]

    if op == "Unsqueeze":
        x = get(0)
        axes = _attr(node, "axes", None)
        if axes is None:
            axes = _np(get(1)).tolist()
        for ax in sorted([int(a) for a in axes]):
            x = F.expand_dims(x, axis=ax)
        return [x]

    if op == "Squeeze":
        x = get(0)
        axes_t = get(1)
        axes = _attr(node, "axes", None)
        if axes is None and axes_t is not None:
            axes = _np(axes_t).tolist()
        if axes:
            for ax in sorted([int(a) for a in axes], reverse=True):
                x = F.squeeze(x, axis=ax)
        else:
            x = F.squeeze(x)
        return [x]

    if op == "Expand":
        x = get(0)
        shape = _np(get(1)).tolist()
        return [F.broadcast_to(x, shape)]

    if op == "Gather":
        x   = get(0)
        idx = _np(get(1))          # indices always come from initializers
        axis = int(_attr(node, "axis", 0))
        if _is_jax_module(F):
            return [F.take(x, idx.astype(np.int32), axis=axis)]
        else:
            import tensorflow as tf
            return [tf.gather(x, idx.astype(np.int32), axis=axis)]

    if op == "Concat":
        axis = int(_attr(node, "axis", 0))
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        return [F.concatenate(tensors, axis=axis)]

    if op == "Split":
        x = get(0)
        axis = int(_attr(node, "axis", 0))
        split_t = get(1)
        sizes = _attr(node, "split", None)
        if sizes is None and split_t is not None:
            sizes = _np(split_t).tolist()
        if sizes is None:
            n = len([o for o in node.output if o])
            sizes = [x.shape[axis] // n] * n
        sizes_int = [int(s) for s in sizes]
        indices = np.cumsum(sizes_int[:-1]).tolist()
        # jax.numpy.split uses indices; tf.split uses sizes
        if _is_jax_module(F):
            parts = F.split(x, [int(i) for i in indices], axis=axis)
        else:
            import tensorflow as tf
            parts = tf.split(x, sizes_int, axis=axis)
        return list(parts)

    if op == "Slice":
        x = get(0)
        starts  = _np(get(1)).tolist()
        ends    = _np(get(2)).tolist()
        axes_t  = get(3); steps_t = get(4)
        axes  = _np(axes_t).tolist() if axes_t is not None else list(range(len(starts)))
        steps = _np(steps_t).tolist() if steps_t is not None else [1]*len(starts)
        slices = [slice(None)] * len(x.shape)
        for ax, s, e, st in zip(axes, starts, ends, steps):
            ax = int(ax) % len(x.shape)
            slices[ax] = slice(int(s), int(e) if abs(int(e)) < 2**30 else None, int(st))
        return [x[tuple(slices)]]

    if op == "Pad":
        x = get(0)
        pads_t = get(1)
        pads = _attr(node, "pads", None)
        if pads is None:
            pads = _np(pads_t).tolist()
        mode = _attr(node, "mode", b"constant")
        if isinstance(mode, bytes): mode = mode.decode()
        n = len(x.shape)
        pad_width = [(int(pads[i]), int(pads[i+n])) for i in range(n)]
        if _is_jax_module(F):
            import jax.numpy as jnp
            return [jnp.pad(x, pad_width, mode=mode if mode != "constant" else "constant")]
        else:
            import tensorflow as tf
            paddings = tf.constant(pad_width, dtype=tf.int32)
            return [tf.pad(x, paddings)]

    if op == "Tile":
        x = get(0)
        reps = _np(get(1)).tolist()
        return [F.tile(x, [int(r) for r in reps])]

    # ── Linear algebra ───────────────────────────────────────────────────────
    if op == "MatMul":
        return [F.matmul(get(0), get(1))]

    if op == "Gemm":
        A = get(0); B = get(1); C = get(2)
        alpha = float(_attr(node, "alpha", 1.0))
        beta  = float(_attr(node, "beta",  1.0))
        if _attr(node, "transA", 0): A = F.swapaxes(A, -1, -2) if hasattr(F,'swapaxes') else F.transpose(A, list(range(len(A.shape)-2))+[-1,-2])
        if _attr(node, "transB", 0): B = F.swapaxes(B, -1, -2) if hasattr(F,'swapaxes') else F.transpose(B, list(range(len(B.shape)-2))+[-1,-2])
        result = np.float32(alpha) * F.matmul(A, B)
        if C is not None:
            result = result + np.float32(beta) * C
        return [result]

    # ── Convolution ──────────────────────────────────────────────────────────
    if op == "Conv":
        return _conv(node, get, F)

    if op == "ConvTranspose":
        return _conv_transpose(node, get, F)

    # ── Normalization ────────────────────────────────────────────────────────
    if op == "BatchNormalization":
        x = get(0); scale = get(1); B_ = get(2); mean = get(3); var = get(4)
        eps = float(_attr(node, "epsilon", 1e-5))
        nd = len(x.shape) - 2
        bc = [1, -1] + [1]*nd
        x_n = (x - F.reshape(mean, bc)) / F.sqrt(F.reshape(var, bc) + np.float32(eps))
        return [F.reshape(scale, bc) * x_n + F.reshape(B_, bc)]

    if op == "InstanceNormalization":
        x = get(0); scale = get(1); B_ = get(2)
        eps = float(_attr(node, "epsilon", 1e-5))
        axes = tuple(range(2, len(x.shape)))
        nd = len(x.shape) - 2
        bc = [1, -1] + [1]*nd
        mean = F.mean(x, axis=axes, keepdims=True)
        var  = F.mean((x-mean)**2, axis=axes, keepdims=True)
        x_n  = (x - mean) / F.sqrt(var + np.float32(eps))
        return [F.reshape(scale, bc) * x_n + F.reshape(B_, bc)]

    if op == "LayerNormalization":
        x = get(0); scale = get(1); B_ = get(2)
        axis = int(_attr(node, "axis", -1))
        eps  = float(_attr(node, "epsilon", 1e-5))
        ndim = len(x.shape)
        norm_axis = axis % ndim
        axes = tuple(range(norm_axis, ndim))
        mean = F.mean(x, axis=axes, keepdims=True)
        var  = F.mean((x - mean)**2, axis=axes, keepdims=True)
        x_n  = (x - mean) / F.sqrt(var + np.float32(eps))
        if scale is not None: x_n = x_n * scale
        if B_ is not None:    x_n = x_n + B_
        return [x_n]

    # ── Pooling ──────────────────────────────────────────────────────────────
    if op in ("MaxPool", "AveragePool"):
        return _pool(node, get, F)

    if op == "GlobalAveragePool":
        x = get(0)
        axes = tuple(range(2, len(x.shape)))
        return [F.mean(x, axis=axes, keepdims=True)]

    if op == "GlobalMaxPool":
        x = get(0)
        axes = tuple(range(2, len(x.shape)))
        return [F.max(x, axis=axes, keepdims=True)]

    # ── Reductions ───────────────────────────────────────────────────────────
    if op == "ReduceMean":
        x = get(0)
        axes = _attr(node, "axes", None)
        if axes is None:
            at = get(1)
            if at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.mean(x, axis=ax, keepdims=kd)]

    if op == "ReduceSum":
        x = get(0)
        at = get(1); axes = _attr(node, "axes", None)
        if axes is None and at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.sum(x, axis=ax, keepdims=kd)]

    if op == "ReduceMax":
        x = get(0)
        axes = _attr(node, "axes", None)
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.max(x, axis=ax, keepdims=kd)]

    if op == "ReduceL2":
        x = get(0)
        at = get(1); axes = _attr(node, "axes", None)
        if axes is None and at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.sqrt(F.sum(x*x, axis=ax, keepdims=kd))]

    # ── Misc ─────────────────────────────────────────────────────────────────
    if op == "Where":
        return [F.where(get(0), get(1), get(2))]

    if op == "DepthToSpace":
        x = get(0)
        bs   = int(_attr(node, "blocksize", 2))
        mode = _attr(node, "mode", b"DCR")
        if isinstance(mode, bytes): mode = mode.decode()
        N, C, H, W = x.shape
        if mode == "DCR":
            x = F.reshape(x, [N, bs, bs, C//(bs*bs), H, W])
            x = F.transpose(x, [0, 3, 4, 1, 5, 2])
        else:  # CRD
            x = F.reshape(x, [N, C//(bs*bs), bs, bs, H, W])
            x = F.transpose(x, [0, 1, 4, 2, 5, 3])
        return [F.reshape(x, [N, C//(bs*bs), H*bs, W*bs])]

    if op == "Resize":
        return _resize(node, get, F)

    if op == "ConstantOfShape":
        shape = _np(get(0)).tolist()
        val_attr = _attr(node, "value", None)
        val = float(val_attr.flat[0]) if val_attr is not None else 0.0
        return [F.full(shape, np.float32(val)) if hasattr(F,'full')
                else np.full(shape, np.float32(val))]

    if op == "Shape":
        x = get(0)
        return [np.array(x.shape, dtype=np.int64)]

    if op == "Reciprocal":
        return [np.float32(1.0) / get(0)]

    if op in ("Equal", "Less", "Greater", "Not", "LessOrEqual", "GreaterOrEqual"):
        a, b = get(0), get(1)
        if op == "Equal":         return [a == b]
        if op == "Less":          return [a < b]
        if op == "Greater":       return [a > b]
        if op == "LessOrEqual":   return [a <= b]
        if op == "GreaterOrEqual":return [a >= b]
        if op == "Not":           return [~get(0)]

    if op == "CumSum":
        x = get(0)
        axis = int(_np(get(1)).flat[0])
        return [F.cumsum(x, axis=axis) if hasattr(F, 'cumsum') else F.cumulative_sum(x, axis=axis)]

    raise NotImplementedError(f"Unsupported ONNX op: {op}")


# ── Framework detection ───────────────────────────────────────────────────────

def _is_jax_module(F) -> bool:
    """Return True if F is jax.numpy (not tf.experimental.numpy)."""
    try:
        import jax.numpy as _jnp
        return F is _jnp
    except ImportError:
        return False


# ── Conv helper ──────────────────────────────────────────────────────────────

def _conv(node, get, F):
    x = get(0); w = get(1); b = get(2)
    pads      = _attr(node, "pads",      [0,0,0,0])
    strides   = _attr(node, "strides",   [1,1])
    dilations = _attr(node, "dilations", [1,1])
    group     = int(_attr(node, "group", 1))

    if _is_jax_module(F):
        import jax.lax as lax
        dn = lax.conv_dimension_numbers(x.shape, w.shape, ("NCHW","OIHW","NCHW"))
        padding = ((int(pads[0]), int(pads[2])), (int(pads[1]), int(pads[3])))
        y = lax.conv_general_dilated(
            x, w,
            window_strides=[int(s) for s in strides],
            padding=padding,
            lhs_dilation=(1,1),
            rhs_dilation=[int(d) for d in dilations],
            dimension_numbers=dn,
            feature_group_count=group,
        )
    else:
        import tensorflow as tf
        # TF conv: NHWC format
        x_nhwc = tf.transpose(x, [0,2,3,1])
        w_hwio = tf.transpose(w, [2,3,1,0])  # OIHW → HWIO
        if group == 1:
            y_nhwc = tf.nn.conv2d(
                x_nhwc, w_hwio,
                strides=[1, int(strides[0]), int(strides[1]), 1],
                padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
                dilations=[int(dilations[0]), int(dilations[1])],
            )
        else:
            # depthwise conv: w is [C,1,kH,kW] → need [kH,kW,C,1] for tf
            w_dwconv = tf.transpose(w, [2,3,0,1])  # [kH,kW,C,1]
            y_nhwc = tf.nn.depthwise_conv2d(
                x_nhwc, w_dwconv,
                strides=[1, int(strides[0]), int(strides[1]), 1],
                padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
                dilations=[int(dilations[0]), int(dilations[1])],
            )
        y = tf.transpose(y_nhwc, [0,3,1,2])

    if b is not None:
        y = y + F.reshape(b, [1,-1,1,1])
    return [y]


# ── ConvTranspose helper ──────────────────────────────────────────────────────

def _conv_transpose(node, get, F):
    x = get(0); w = get(1); b = get(2)
    pads      = _attr(node, "pads",      [0,0,0,0])
    strides   = _attr(node, "strides",   [1,1])
    dilations = _attr(node, "dilations", [1,1])
    op_pads   = _attr(node, "output_padding", [0,0])
    group     = int(_attr(node, "group", 1))

    if _is_jax_module(F):
        import jax.lax as lax
        # ONNX ConvTranspose: w is [C_in, C_out/group, kH, kW].
        # Implement as dilated conv (lhs_dilation = strides) with spatially-flipped
        # and transposed weight → [C_out, C_in, kH, kW] in OIHW format.
        # Padding: for each spatial dim, pad = kernel - 1 - original_pad.
        kH = int(w.shape[2]); kW = int(w.shape[3])
        sH = int(strides[0]); sW = int(strides[1])
        dH = int(dilations[0]); dW = int(dilations[1])
        # Effective kernel size with dilation
        ekH = dH * (kH - 1) + 1; ekW = dW * (kW - 1) + 1
        # Transpose weight: [C_in, C_out, kH, kW] → [C_out, C_in, kH, kW], flip spatially
        w_t = F.transpose(w, (1, 0, 2, 3))[:, :, ::-1, ::-1]
        pad_h_top = ekH - 1 - int(pads[0]); pad_h_bot = ekH - 1 - int(pads[2]) + int(op_pads[0])
        pad_w_left = ekW - 1 - int(pads[1]); pad_w_right = ekW - 1 - int(pads[3]) + int(op_pads[1])
        y = lax.conv_general_dilated(
            x, w_t,
            window_strides=(1, 1),
            padding=((pad_h_top, pad_h_bot), (pad_w_left, pad_w_right)),
            lhs_dilation=(sH, sW),
            rhs_dilation=(dH, dW),
            feature_group_count=group,
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        )
    else:
        import tensorflow as tf
        # For ConvTranspose: w is [C_in, C_out/group, kH, kW] in ONNX
        # TF conv2d_transpose expects [kH, kW, C_out, C_in]
        x_nhwc = tf.transpose(x, [0,2,3,1])
        N, H_in, W_in, C_in = [int(d) for d in x_nhwc.shape]
        C_out = int(w.shape[1]) * group
        kH, kW = int(w.shape[2]), int(w.shape[3])
        sH, sW = int(strides[0]), int(strides[1])
        H_out = (H_in - 1) * sH - int(pads[0]) - int(pads[2]) + kH + int(op_pads[0])
        W_out = (W_in - 1) * sW - int(pads[1]) - int(pads[3]) + kW + int(op_pads[1])
        w_tf = tf.transpose(w, [2,3,1,0])  # [kH,kW,C_out/g,C_in]
        output_shape = [N, H_out, W_out, C_out]
        y_nhwc = tf.nn.conv2d_transpose(
            x_nhwc, w_tf,
            output_shape=output_shape,
            strides=[1, sH, sW, 1],
            padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
        )
        y = tf.transpose(y_nhwc, [0,3,1,2])

    if b is not None:
        y = y + F.reshape(b, [1,-1,1,1])
    return [y]


# ── Pool helper ──────────────────────────────────────────────────────────────

def _pool(node, get, F):
    op = node.op_type
    x = get(0)
    k         = _attr(node, "kernel_shape", [2,2])
    strides   = _attr(node, "strides",      [1,1])
    pads      = _attr(node, "pads",         [0,0,0,0])
    dilations = _attr(node, "dilations",    [1,1])
    ceil_mode = int(_attr(node, "ceil_mode", 0))

    dH, dW = int(dilations[0]), int(dilations[1])

    if _is_jax_module(F):
        import jax.lax as lax
        import jax.numpy as jnp
        pH0, pH1 = int(pads[0]), int(pads[2])
        pW0, pW1 = int(pads[1]), int(pads[3])
        if ceil_mode == 1:
            # Add extra right/bottom padding so lax.reduce_window matches ceil-mode output size
            in_H = int(x.shape[2]); in_W = int(x.shape[3])
            sH = int(strides[0]);   sW = int(strides[1])
            ekH = dH * (int(k[0]) - 1) + 1; ekW = dW * (int(k[1]) - 1) + 1
            rem_h = (in_H + pH0 + pH1 - ekH) % sH
            rem_w = (in_W + pW0 + pW1 - ekW) % sW
            pH1 += (sH - rem_h) if rem_h != 0 else 0
            pW1 += (sW - rem_w) if rem_w != 0 else 0
        pad_h = (pH0, pH1); pad_w = (pW0, pW1)
        padding = ((0,0),(0,0), pad_h, pad_w)
        window = (1, 1, int(k[0]), int(k[1]))
        str_   = (1, 1, int(strides[0]), int(strides[1]))
        win_dil = (1, 1, dH, dW)
        if op == "MaxPool":
            y = lax.reduce_window(x, -jnp.inf, lax.max, window, str_, padding,
                                  window_dilation=win_dil)
        else:
            ones = F.ones_like(x)
            s = lax.reduce_window(x,    0.0, lax.add, window, str_, padding,
                                  window_dilation=win_dil)
            n = lax.reduce_window(ones, 0.0, lax.add, window, str_, padding,
                                  window_dilation=win_dil)
            y = s / n
    else:
        import tensorflow as tf
        x_nhwc = tf.transpose(x, [0,2,3,1])
        ksize   = [1, int(k[0]),       int(k[1]),       1]
        str_tf  = [1, int(strides[0]), int(strides[1]), 1]
        paddings_tf = [[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]]
        if op == "MaxPool" and (dH > 1 or dW > 1):
            # TF max_pool2d has no dilation support; use extract_patches + reduce_max
            kH, kW = int(k[0]), int(k[1])
            pH0, pH1 = int(pads[0]), int(pads[2])
            pW0, pW1 = int(pads[1]), int(pads[3])
            x_pad = tf.pad(x_nhwc, [[0,0],[pH0,pH1],[pW0,pW1],[0,0]],
                           constant_values=-1e9)
            patches = tf.image.extract_patches(
                x_pad,
                sizes=[1, kH, kW, 1],
                strides=str_tf,
                rates=[1, dH, dW, 1],
                padding="VALID",
            )
            N_, H_out, W_out, C_ = [int(d) for d in x_nhwc.shape]
            H_out2 = patches.shape[1]; W_out2 = patches.shape[2]
            C_in = int(x_nhwc.shape[-1])
            patches_r = tf.reshape(patches, [-1, H_out2, W_out2, kH * kW, C_in])
            y_nhwc = tf.reduce_max(patches_r, axis=3)
        elif op == "MaxPool":
            y_nhwc = tf.nn.max_pool2d(x_nhwc, ksize, str_tf, padding=paddings_tf)
        else:
            # avg_pool doesn't support dilations in TF, treat as no dilation
            y_nhwc = tf.nn.avg_pool2d(x_nhwc, ksize, str_tf, padding=paddings_tf)
        y = tf.transpose(y_nhwc, [0,3,1,2])
    return [y]


# ── Resize helper ─────────────────────────────────────────────────────────────

def _resize(node, get, F):
    x = get(0)
    scales_t = get(2); sizes_t = get(3)
    mode = _attr(node, "mode", b"nearest")
    if isinstance(mode, bytes): mode = mode.decode()
    N, C = int(x.shape[0]), int(x.shape[1])
    if scales_t is not None:
        scales = _np(scales_t).tolist()
        H_new = int(int(x.shape[2]) * scales[2])
        W_new = int(int(x.shape[3]) * scales[3])
    else:
        ts = _np(sizes_t).tolist()
        H_new, W_new = int(ts[2]), int(ts[3])

    if _is_jax_module(F):
        import jax.image as ji
        import jax.numpy as jnp
        x_nhwc = jnp.transpose(x, (0,2,3,1))
        method = "nearest" if "nearest" in mode else "linear"
        y_nhwc = ji.resize(x_nhwc, (N, H_new, W_new, C), method=method)
        return [jnp.transpose(y_nhwc, (0,3,1,2))]
    else:
        import tensorflow as tf
        x_nhwc = tf.transpose(x, [0,2,3,1])
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR if "nearest" in mode \
                 else tf.image.ResizeMethod.BILINEAR
        y_nhwc = tf.image.resize(x_nhwc, [H_new, W_new], method=method)
        return [tf.transpose(y_nhwc, [0,3,1,2])]
