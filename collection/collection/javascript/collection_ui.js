
onUiLoaded(() => {
    if (window.collectionInfiniteScrollInit) return;
    window.collectionInfiniteScrollInit = true;

    let observer = null;

    function getRoot() {
        if (typeof gradioApp === "function") return gradioApp();
        return document;
    }

    function getSelectedCollectionIdInput(root) {
        return root.querySelector("#collection_selected_collection_id textarea, #collection_selected_collection_id input");
    }

    function getBatchOffsetInput(root) {
        return root.querySelector("#collection_batch_offset textarea, #collection_batch_offset input");
    }

    function getLoadMoreButton(root) {
        return root.querySelector("#collection_load_more_button button, #collection_load_more_button");
    }

    function processBatchPayload() {
        const root = getRoot();
        const payloadHost = root.querySelector("#collection_batch_payload");
        if (!payloadHost) return;

        const payloadEl = payloadHost.querySelector(".collection-batch-payload");
        if (!payloadEl || payloadEl.dataset.processed === "1") return;

        payloadEl.dataset.processed = "1";

        let data = null;
        try {
            data = JSON.parse(payloadEl.getAttribute("data-json") || "{}");
        } catch (err) {
            console.error("collection batch payload parse failed", err);
            return;
        }

        const cardsContainer = root.querySelector("#collection_cards_container");
        const sentinel = root.querySelector("#collection_feed_sentinel");

        if (cardsContainer && data.html) {
            const temp = document.createElement("div");
            temp.innerHTML = data.html;

            // Append nodes
            cardsContainer.append(...temp.children);
        }    

        if (sentinel) {
            sentinel.dataset.loading = "0";
            sentinel.dataset.nextOffset = String(data.next_offset ?? 0);
            sentinel.dataset.hasMore = data.has_more ? "true" : "false";
            sentinel.textContent = data.has_more ? "Loading more..." : "End of collection";
        }
    }

    function ensureInfiniteObserver() {
        const root = getRoot();
        const sentinel = root.querySelector("#collection_feed_sentinel");
        const selectedCollectionInput = getSelectedCollectionIdInput(root);

        if (!sentinel || !selectedCollectionInput || !selectedCollectionInput.value) return;

        if (sentinel.dataset.observed === "1") return;
        sentinel.dataset.observed = "1";

        if (observer) observer.disconnect();

        observer = new IntersectionObserver(
            (entries) => {
                for (const entry of entries) {
                    if (!entry.isIntersecting) continue;

                    const target = entry.target;

                    if (target.dataset.loading === "1") return;
                    if (target.dataset.hasMore !== "true") return;

                    const offsetInput = getBatchOffsetInput(root);
                    const triggerButton = getLoadMoreButton(root);

                    if (!offsetInput || !triggerButton) return;

                    target.dataset.loading = "1";

                    offsetInput.value = target.dataset.nextOffset || "0";
                    offsetInput.dispatchEvent(new Event("input", { bubbles: true }));
                    offsetInput.dispatchEvent(new Event("change", { bubbles: true }));

                    triggerButton.click();
                }
            },
            {
                root: null,
                rootMargin: "900px 0px 900px 0px",
                threshold: 0.01
            }
        );

        observer.observe(sentinel);
    }


    setInterval(processBatchPayload, 250);
    setInterval(ensureInfiniteObserver, 400);
});